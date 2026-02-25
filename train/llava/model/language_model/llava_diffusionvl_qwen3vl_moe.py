# coding=utf-8
# Copyright 2025 The HustVL Team and The HuggingFace Inc. team. All rights reserved.
#
# This code extends DiffusionVL to support Qwen3-VL-MoE with BD3-LM training.

"""DiffusionVL-Qwen3VL-MoE model implementation."""

import json
import math
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig as HFPretrainedConfig
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeVisionConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3VLMoePreTrainedModel,
    Qwen3VLMoeTextAttention,
    Qwen3VLMoeTextModel as Qwen3VLMoeTextModelOriginal,
    Qwen3VLMoeTextSparseMoeBlock,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from llava.constants import IGNORE_INDEX
from llava.model.llava_arch import LlavaMetaForCausalLM, LlavaMetaModel
from llava.utils import rank0_print

logger = logging.get_logger(__name__)


class DiffusionVLQwen3VLMoeConfig(HFPretrainedConfig):
    """Configuration class for DiffusionVL-Qwen3VL-MoE model."""

    model_type = "diffusionvl_qwen3vl_moe"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=151655,
        video_token_id=151656,
        enable_bd3lm=False,
        bd3lm_block_size=4,
        # Module 1: NCR
        enable_ncr=False,
        ncr_fourier_dim=16,
        # Module 2: BEBC
        enable_bebc=False,
        lambda_lb=0.0,
        lambda_scr=0.0,
        scr_ema_decay=0.999,
        scr_stage1_steps=2000,
        scr_stage2_steps=8000,
        # Module 3: DSA
        enable_dsa=False,
        lambda_dsa=0.0,
        dsa_temperature=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
        if text_config is None:
            text_config = {}

        self.vision_config = Qwen3VLMoeVisionConfig(**vision_config)
        self.text_config = Qwen3VLMoeTextConfig(**text_config)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        self.enable_bd3lm = enable_bd3lm
        self.bd3lm_block_size = bd3lm_block_size

        self.enable_ncr = enable_ncr
        self.ncr_fourier_dim = ncr_fourier_dim

        self.enable_bebc = enable_bebc
        self.lambda_lb = lambda_lb
        self.lambda_scr = lambda_scr
        self.scr_ema_decay = scr_ema_decay
        self.scr_stage1_steps = scr_stage1_steps
        self.scr_stage2_steps = scr_stage2_steps

        self.enable_dsa = enable_dsa
        self.lambda_dsa = lambda_dsa
        self.dsa_temperature = dsa_temperature

        if self.enable_bd3lm:
            self.bd3lm_antithetic_sampling = True
            self.bd3lm_sampling_eps_min = 1e-3
            self.bd3lm_sampling_eps_max = 1.0

        for key, value in self.text_config.to_dict().items():
            setattr(self, key, value)
        # Keep diffusion MoE checkpoint loading deterministic: do not tie embeddings/lm_head.
        # This model carries extra newly-initialized routing heads (NCR) and tying can corrupt shapes.
        self.tie_word_embeddings = False

    def to_dict(self):
        output = super().to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        return output


class NoiseConditionedSparseMoeBlock(Qwen3VLMoeTextSparseMoeBlock):
    """Sparse MoE block with optional NCR features and router cache for BEBC."""

    def __init__(self, config):
        super().__init__(config)
        self.enable_ncr = bool(getattr(config, "enable_ncr", False))
        self.ncr_fourier_dim = int(getattr(config, "ncr_fourier_dim", 16))
        self.hidden_size = config.hidden_size
        self.top_k = config.num_experts_per_tok

        if self.enable_ncr:
            fourier_out_dim = self.ncr_fourier_dim * 2
            self.t_proj = nn.Linear(fourier_out_dim, self.hidden_size, bias=False)
            self.m_proj = nn.Linear(1, self.hidden_size, bias=False)

        self._router_t: Optional[torch.Tensor] = None
        self._router_m: Optional[torch.Tensor] = None
        self._router_shape: Optional[Tuple[int, int]] = None
        self._last_router_probs: Optional[torch.Tensor] = None
        self._last_topk_indices: Optional[torch.Tensor] = None

    def set_conditioning(self, router_t: Optional[torch.Tensor], router_m: Optional[torch.Tensor]):
        self._router_t = router_t
        self._router_m = router_m
        if router_t is not None:
            self._router_shape = (router_t.shape[0], router_t.shape[1])
        else:
            self._router_shape = None

    def clear_cache(self):
        self._last_router_probs = None
        self._last_topk_indices = None

    def _fourier_embedding(self, t: torch.Tensor) -> torch.Tensor:
        if self.ncr_fourier_dim <= 0:
            return t.unsqueeze(-1)
        half_dim = self.ncr_fourier_dim
        freq = torch.arange(half_dim, device=t.device, dtype=t.dtype)
        freq = torch.exp(-math.log(10000.0) * (freq / max(half_dim - 1, 1)))
        x = t.unsqueeze(-1) * freq.unsqueeze(0)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(-1, self.hidden_size)
        routing_hidden = hidden_states_flat

        if self.enable_ncr and self._router_t is not None and self._router_m is not None:
            flat_t = self._router_t.reshape(-1).to(hidden_states_flat.device, hidden_states_flat.dtype)
            flat_m = self._router_m.reshape(-1).to(hidden_states_flat.device, hidden_states_flat.dtype)
            if flat_t.numel() == hidden_states_flat.shape[0] and flat_m.numel() == hidden_states_flat.shape[0]:
                fourier = self._fourier_embedding(flat_t)
                t_weight_shape = tuple(self.t_proj.weight.shape) if hasattr(self, "t_proj") and hasattr(self.t_proj, "weight") else None
                m_weight_shape = tuple(self.m_proj.weight.shape) if hasattr(self, "m_proj") and hasattr(self.m_proj, "weight") else None
                proj_ok = (
                    hasattr(self, "t_proj")
                    and hasattr(self, "m_proj")
                    and t_weight_shape == (hidden_states_flat.shape[-1], fourier.shape[-1])
                    and m_weight_shape == (hidden_states_flat.shape[-1], 1)
                )
                if not proj_ok:
                    raise RuntimeError(
                        "NCR projection shape mismatch in NoiseConditionedSparseMoeBlock: "
                        f"expected t_proj.weight {(hidden_states_flat.shape[-1], fourier.shape[-1])}, got {t_weight_shape}; "
                        f"expected m_proj.weight {(hidden_states_flat.shape[-1], 1)}, got {m_weight_shape}."
                    )

                t_feat = self.t_proj(fourier)
                m_feat = self.m_proj(flat_m.unsqueeze(-1))
                routing_hidden = routing_hidden + t_feat + m_feat

        router_probs = F.linear(routing_hidden, self.gate.weight)
        router_probs = F.softmax(router_probs, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states_flat.dtype)

        routed_out = self.experts(hidden_states_flat, router_indices, routing_weights)

        self._last_router_probs = router_probs
        self._last_topk_indices = router_indices
        return routed_out.reshape(batch_size, sequence_length, self.hidden_size)


class DiffusionVLQwen3VLMoeAttention(Qwen3VLMoeTextAttention):
    """Attention layer with non-causal mask for BD3-LM diffusion."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        store_kv = kwargs.pop("store_kv", True)
        flash_attn_kwargs = kwargs

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            if store_kv:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                past_key_states = None
                past_value_states = None
                if hasattr(past_key_values, "layers"):
                    if self.layer_idx < len(past_key_values.layers):
                        layer_cache = past_key_values.layers[self.layer_idx]
                        past_key_states = getattr(layer_cache, "keys", None)
                        past_value_states = getattr(layer_cache, "values", None)
                elif self.layer_idx < len(past_key_values):
                    past_key_states, past_value_states = past_key_values[self.layer_idx]

                if past_key_states is not None and past_value_states is not None:
                    key_states = torch.cat([past_key_states, key_states], dim=2)
                    value_states = torch.cat([past_value_states, value_states], dim=2)

        if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation,
                eager_attention_forward,
            )
        else:
            attention_interface = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None),
            position_ids=position_ids,
            **flash_attn_kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class DiffusionVLQwen3VLMoeTextModel(Qwen3VLMoeTextModelOriginal):
    """Qwen3 text model with diffusion-compatible attention and router hooks."""

    def __init__(self, config):
        super().__init__(config)

        for layer in self.layers:
            original_layer_idx = layer.self_attn.layer_idx
            layer.self_attn = DiffusionVLQwen3VLMoeAttention(config, layer_idx=original_layer_idx)

            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, Qwen3VLMoeTextSparseMoeBlock):
                layer.mlp = NoiseConditionedSparseMoeBlock(config)

        self._router_cache: List[Dict[str, torch.Tensor]] = []

        if getattr(config, "enable_bd3lm", False):
            self._init_bd3lm_components(config)

    def _init_bd3lm_components(self, config):
        from .bd3lm_utils import LogLinearNoise

        self.noise_scheduler = LogLinearNoise()
        self.mask_token_id = 151671
        self.bd3lm_block_size = config.bd3lm_block_size
        self.antithetic_sampling = getattr(config, "bd3lm_antithetic_sampling", True)
        self.sampling_eps_min = getattr(config, "bd3lm_sampling_eps_min", 1e-3)
        self.sampling_eps_max = getattr(config, "bd3lm_sampling_eps_max", 1.0)

    def _set_router_conditioning(self):
        router_t = getattr(self, "_bd3lm_router_t", None)
        router_m = getattr(self, "_bd3lm_router_m", None)

        for layer in self.layers:
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, NoiseConditionedSparseMoeBlock):
                mlp.clear_cache()
                mlp.set_conditioning(router_t, router_m)

    def _collect_router_cache(self):
        router_cache = []
        for layer_idx, layer in enumerate(self.layers):
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, NoiseConditionedSparseMoeBlock) and mlp._last_router_probs is not None:
                router_cache.append(
                    {
                        "layer_idx": layer_idx,
                        "router_probs": mlp._last_router_probs,
                        "topk_indices": mlp._last_topk_indices,
                        "router_shape": mlp._router_shape,
                    }
                )
        self._router_cache = router_cache

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sample_mode: Optional[bool] = False,
        store_kv: Optional[bool] = False,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs["store_kv"] = store_kv

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
            causal_mask_mapping = {
                "full_attention": attention_mask,
                "sliding_attention": attention_mask if getattr(self, "has_sliding_layers", False) else None,
            }

        if cache_position is None:
            past_seen_tokens = past_key_values_length
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

        bsz = inputs_embeds.shape[0]
        if position_ids is None:
            text_pos = cache_position.view(1, -1).expand(bsz, -1)
            position_ids = text_pos.unsqueeze(0).expand(4, bsz, -1)
        elif position_ids is not None and position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids
        else:
            text_position_ids = position_ids[0]
            rope_position_ids = text_position_ids[None, ...].expand(3, text_position_ids.shape[0], -1)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)

        self._set_router_conditioning()

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_type = getattr(decoder_layer, "attention_type", None)
            if layer_type == "sliding_attention":
                layer_mask = causal_mask_mapping["sliding_attention"]
            else:
                layer_mask = causal_mask_mapping["full_attention"]

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, (tuple, list)) else layer_outputs
            if output_attentions and isinstance(layer_outputs, (tuple, list)) and len(layer_outputs) > 1:
                all_self_attns += (layer_outputs[1],)

        self._collect_router_cache()

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DiffusionVLQwen3VLMoeForCausalLM_Base(Qwen3VLMoePreTrainedModel):
    """Base CausalLM model with BD3-LM, NCR, BEBC and optional DSA loss."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = DiffusionVLQwen3VLMoeTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.enable_bebc = bool(getattr(config, "enable_bebc", False))
        self.lambda_lb = float(getattr(config, "lambda_lb", 0.0))
        self.lambda_scr = float(getattr(config, "lambda_scr", 0.0))
        self.scr_ema_decay = float(getattr(config, "scr_ema_decay", 0.999))
        self.scr_stage1_steps = int(getattr(config, "scr_stage1_steps", 2000))
        self.scr_stage2_steps = int(getattr(config, "scr_stage2_steps", 8000))

        self.enable_dsa = bool(getattr(config, "enable_dsa", False))
        self.lambda_dsa = float(getattr(config, "lambda_dsa", 0.0))
        self.dsa_temperature = float(getattr(config, "dsa_temperature", 1.0))

        self._router_ema: Dict[str, torch.Tensor] = {}
        self._forward_calls = 0

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _apply_bd3lm_noise_embedding(self, inputs_embeds, labels):
        """Apply BD3-LM noise in embedding space."""
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        block_size = self.model.bd3lm_block_size
        num_blocks = (seq_len + block_size - 1) // block_size
        _eps_b = torch.rand((batch_size, num_blocks), device=device)

        if self.model.antithetic_sampling:
            num_samples = _eps_b.numel()
            offset = torch.arange(num_samples, device=device) / num_samples
            offset = offset.view(_eps_b.shape)
            _eps_b = (_eps_b / num_samples + offset) % 1

        t = _eps_b.repeat_interleave(block_size, dim=-1)
        t = t[:, :seq_len]
        t = t * (self.model.sampling_eps_max - self.model.sampling_eps_min) + self.model.sampling_eps_min

        loss_scale, p = self.model.noise_scheduler(t)

        move_probabilities = torch.rand(batch_size, seq_len, device=device)
        move_chance = p
        text_token_mask = labels != IGNORE_INDEX
        move_indices = (move_probabilities <= move_chance) & text_token_mask

        mask_embed = self.get_input_embeddings()(torch.tensor([151671], device=device))
        xt_embeds = torch.where(move_indices.unsqueeze(-1), mask_embed, inputs_embeds)

        avg_noise_level = torch.mean(move_chance).item()
        bd3lm_inputs = torch.cat([xt_embeds, inputs_embeds], dim=1)

        # NCR conditioning for concatenated [x_t, x_0]
        zeros = torch.zeros_like(t)
        router_t = torch.cat([t, zeros], dim=1)
        router_m = torch.cat([move_indices.to(t.dtype), torch.zeros_like(t)], dim=1)
        self.model._bd3lm_router_t = router_t
        self.model._bd3lm_router_m = router_m

        return bd3lm_inputs, move_indices, loss_scale, inputs_embeds, avg_noise_level

    def _compute_bd3lm_loss_embedding(self, logits, labels, move_indices, loss_scale):
        """Compute BD3-LM loss and per-sample losses for optional DSA."""
        masked_positions = move_indices & (labels != IGNORE_INDEX)
        batch_size = labels.shape[0]

        if not masked_positions.any():
            zero = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return zero, torch.zeros(batch_size, device=logits.device, dtype=logits.dtype)

        logits_flat = logits[masked_positions]
        labels_flat = labels[masked_positions]
        token_loss_unweighted = F.cross_entropy(logits_flat, labels_flat, reduction="none")

        loss_scale_flat = loss_scale[masked_positions]
        weighted_loss = token_loss_unweighted * loss_scale_flat.abs()

        prompt_index = (labels == IGNORE_INDEX).to(torch.int64)
        noisy_data_length = torch.sum((1 - prompt_index), dim=-1, keepdim=True)
        noisy_data_length = torch.max(noisy_data_length, torch.ones_like(noisy_data_length))

        token_loss_full = torch.zeros_like(labels, dtype=logits.dtype)
        token_loss_full[masked_positions] = weighted_loss

        sample_loss = token_loss_full.sum(dim=1) / noisy_data_length.squeeze(-1).to(logits.dtype)
        loss = sample_loss.mean()
        return loss, sample_loss

    def _compute_load_balancing_loss(self):
        if not self.enable_bebc or self.lambda_lb <= 0:
            return torch.tensor(0.0, device=self.lm_head.weight.device)

        router_cache = getattr(self.model, "_router_cache", None)
        if not router_cache:
            return torch.tensor(0.0, device=self.lm_head.weight.device)

        losses = []
        for entry in router_cache:
            probs = entry["router_probs"]  # [N, E]
            topk_indices = entry["topk_indices"]  # [N, K]
            if probs is None or topk_indices is None:
                continue

            assign = torch.zeros_like(probs)
            assign.scatter_(1, topk_indices, 1.0)

            router_prob_per_expert = probs.mean(dim=0)
            tokens_per_expert = assign.mean(dim=0)
            e = probs.shape[1]
            losses.append(torch.sum(tokens_per_expert * router_prob_per_expert) * e)

        if not losses:
            return torch.tensor(0.0, device=self.lm_head.weight.device)
        return torch.stack(losses).mean()

    def _compute_scr_dual_noise_loss(self):
        router_cache = getattr(self.model, "_router_cache", None)
        if not router_cache:
            return torch.tensor(0.0, device=self.lm_head.weight.device)

        losses = []
        eps = 1e-8
        for entry in router_cache:
            probs = entry["router_probs"]
            router_shape = entry.get("router_shape", None)
            if probs is None or router_shape is None:
                continue

            bsz, seq_total = router_shape
            if seq_total % 2 != 0 or probs.shape[0] != bsz * seq_total:
                continue

            probs_3d = probs.view(bsz, seq_total, -1)
            half = seq_total // 2
            p_xt = probs_3d[:, :half, :].reshape(-1, probs_3d.shape[-1])
            p_x0 = probs_3d[:, half:, :].reshape(-1, probs_3d.shape[-1])

            kl_1 = F.kl_div(torch.log(p_xt + eps), p_x0.detach() + eps, reduction="batchmean")
            kl_2 = F.kl_div(torch.log(p_x0 + eps), p_xt.detach() + eps, reduction="batchmean")
            losses.append(0.5 * (kl_1 + kl_2))

        if not losses:
            return torch.tensor(0.0, device=self.lm_head.weight.device)
        return torch.stack(losses).mean()

    def _compute_scr_ema_loss(self):
        router_cache = getattr(self.model, "_router_cache", None)
        if not router_cache:
            return torch.tensor(0.0, device=self.lm_head.weight.device)

        eps = 1e-8
        losses = []
        for entry in router_cache:
            layer_idx = entry["layer_idx"]
            probs = entry["router_probs"]
            if probs is None:
                continue

            cur = probs.mean(dim=0)
            key = f"layer_{layer_idx}"
            if key not in self._router_ema:
                self._router_ema[key] = cur.detach()
                continue

            ema = self._router_ema[key].to(cur.device, cur.dtype)
            losses.append(F.kl_div(torch.log(cur + eps), ema + eps, reduction="batchmean"))
            self._router_ema[key] = self.scr_ema_decay * ema + (1.0 - self.scr_ema_decay) * cur.detach()

        if not losses:
            return torch.tensor(0.0, device=self.lm_head.weight.device)
        return torch.stack(losses).mean()

    def _compute_scr_loss(self):
        if not self.enable_bebc or self.lambda_scr <= 0:
            return torch.tensor(0.0, device=self.lm_head.weight.device)

        # staged interval schedule: 8 -> 4 -> 2
        if self._forward_calls <= self.scr_stage1_steps:
            interval = 8
        elif self._forward_calls <= self.scr_stage2_steps:
            interval = 4
        else:
            interval = 2

        if self._forward_calls % interval != 0:
            return torch.tensor(0.0, device=self.lm_head.weight.device)

        dual_noise = self._compute_scr_dual_noise_loss()
        ema_loss = self._compute_scr_ema_loss()
        return dual_noise + ema_loss

    def _compute_dsa_loss(self, sample_loss, dsa_group_sizes, dsa_answer_idx):
        if not self.enable_dsa or self.lambda_dsa <= 0:
            return torch.tensor(0.0, device=sample_loss.device)
        if dsa_group_sizes is None or dsa_answer_idx is None:
            return torch.tensor(0.0, device=sample_loss.device)

        if isinstance(dsa_group_sizes, torch.Tensor):
            group_sizes = dsa_group_sizes.tolist()
        else:
            group_sizes = list(dsa_group_sizes)

        if isinstance(dsa_answer_idx, torch.Tensor):
            answer_idx = dsa_answer_idx.tolist()
        else:
            answer_idx = list(dsa_answer_idx)

        scores = -sample_loss
        offset = 0
        losses = []
        tau = max(self.dsa_temperature, 1e-6)
        for g, ans in zip(group_sizes, answer_idx):
            if g <= 1:
                offset += g
                continue
            slice_scores = scores[offset: offset + g]
            offset += g
            if slice_scores.numel() != g or ans < 0 or ans >= g:
                continue
            losses.append(F.cross_entropy((slice_scores / tau).unsqueeze(0), torch.tensor([ans], device=scores.device)))

        if not losses:
            return torch.tensor(0.0, device=sample_loss.device)
        return torch.stack(losses).mean()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        dsa_group_sizes: Optional[torch.Tensor] = None,
        dsa_answer_idx: Optional[torch.Tensor] = None,
    ):
        """BD3-LM forward pass with optional NCR/BEBC/DSA losses."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self._forward_calls += 1

        bd3lm_inputs, move_indices, loss_scale, x0_embeds, avg_noise_level = self._apply_bd3lm_noise_embedding(inputs_embeds, labels)

        from .bd3lm_utils import block_diff_mask

        seq_len = inputs_embeds.shape[1]
        device = inputs_embeds.device

        q_idx = torch.arange(seq_len * 2, device=device)[:, None]
        kv_idx = torch.arange(seq_len * 2, device=device)[None, :]

        mask = block_diff_mask(
            b=None,
            h=None,
            q_idx=q_idx,
            kv_idx=kv_idx,
            block_size=self.model.bd3lm_block_size,
            n=seq_len,
        )
        mask = mask.to(torch.bool)

        if attention_mask is not None and attention_mask.dim() == 2:
            extended_attention_mask = torch.cat([attention_mask, attention_mask], dim=1)
            query_validity_mask = extended_attention_mask.unsqueeze(-1)
            key_validity_mask = extended_attention_mask.unsqueeze(-2)
            combined_padding_mask_2d = (query_validity_mask & key_validity_mask).to(torch.bool)
            mask = mask & combined_padding_mask_2d

        attention_mask_4d = torch.zeros(mask.shape, dtype=inputs_embeds.dtype, device=device)
        attention_mask_4d.masked_fill_(~mask, torch.finfo(inputs_embeds.dtype).min)
        attention_mask_4d = attention_mask_4d.unsqueeze(1)

        if position_ids is None:
            pos_ids_part = torch.arange(seq_len, device=device)
            pos_ids = torch.cat([pos_ids_part, pos_ids_part], dim=0)
            position_ids = pos_ids.view(1, -1).expand(inputs_embeds.shape[0], -1)

        outputs = self.model(
            inputs_embeds=bd3lm_inputs,
            attention_mask=attention_mask_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = hidden_states[:, : inputs_embeds.shape[1]]

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        base_loss = torch.tensor(0.0, device=logits.device)
        lb_loss = torch.tensor(0.0, device=logits.device)
        scr_loss = torch.tensor(0.0, device=logits.device)
        dsa_loss = torch.tensor(0.0, device=logits.device)
        sample_loss = None

        if labels is not None:
            base_loss, sample_loss = self._compute_bd3lm_loss_embedding(logits, labels, move_indices, loss_scale)
            loss = base_loss

            if self.enable_bebc:
                lb_loss = self._compute_load_balancing_loss()
                scr_loss = self._compute_scr_loss()
                loss = loss + self.lambda_lb * lb_loss + self.lambda_scr * scr_loss

            if self.enable_dsa and sample_loss is not None:
                dsa_loss = self._compute_dsa_loss(sample_loss, dsa_group_sizes, dsa_answer_idx)
                loss = loss + self.lambda_dsa * dsa_loss

        if self.training:
            if not hasattr(self, "_current_custom_metrics"):
                self._current_custom_metrics = {}
            self._current_custom_metrics["anneal/noise_level"] = avg_noise_level
            self._current_custom_metrics["loss/base_bd3"] = float(base_loss.detach())
            self._current_custom_metrics["loss/lb"] = float(lb_loss.detach())
            self._current_custom_metrics["loss/scr"] = float(scr_loss.detach())
            self._current_custom_metrics["loss/dsa"] = float(dsa_loss.detach())

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate_with_bd3lm(
        self,
        inputs_embeds: torch.FloatTensor,
        steps: int = 4,
        gen_length: int = 128,
        temperature: float = 0.0,
        **kwargs,
    ):
        """BD3-LM inference with KV-cache and block diffusion remasking."""
        from transformers.cache_utils import DynamicCache

        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        prompt_len = inputs_embeds.shape[1]
        block_size = self.model.bd3lm_block_size
        mask_id = 151671

        is_full_diffusion_ablation = block_size >= (prompt_len + gen_length)

        if is_full_diffusion_ablation:
            rank0_print("Full-Diffusion ablation mode enabled.")
            total_length = prompt_len + gen_length
            num_blocks = 1
        else:
            num_blocks = (prompt_len + gen_length + block_size - 1) // block_size
            total_length = num_blocks * block_size

        x_ids = torch.full((batch_size, total_length), mask_id, dtype=torch.long, device=device)
        mask_embed = self.get_input_embeddings()(torch.tensor([mask_id], device=device))
        x_embeds = mask_embed.repeat(batch_size, total_length, 1)
        x_embeds[:, :prompt_len] = inputs_embeds.clone()

        prompt_logits = self.lm_head(inputs_embeds)
        prompt_ids_reconstructed = torch.argmax(prompt_logits, dim=-1)
        x_ids[:, :prompt_len] = prompt_ids_reconstructed

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device)).to(inputs_embeds.dtype)
        block_diffusion_mask_bool = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1).unsqueeze(0)
        block_diffusion_mask = block_diffusion_mask_bool.unsqueeze(1)
        block_diffusion_mask = torch.where(block_diffusion_mask == 0.0, torch.full_like(block_diffusion_mask, float("-inf")), 0.0)
        if is_full_diffusion_ablation:
            block_diffusion_mask = block_diffusion_mask[:, :, :total_length, :total_length]

        position_ids = torch.arange(total_length, device=device).unsqueeze(0).expand(batch_size, -1)

        prefill_blocks = prompt_len // block_size
        prefill_length = prefill_blocks * block_size

        past_key_values = DynamicCache()
        if prefill_length > 0:
            prefill_embeds = x_embeds[:, :prefill_length]
            prefill_mask = block_diffusion_mask[:, :, :prefill_length, :prefill_length]
            prefill_pos_ids = position_ids[:, :prefill_length]
            model_mask = {"full_attention": prefill_mask, "sliding_attention": prefill_mask}

            prefill_outputs = self.model(
                inputs_embeds=prefill_embeds,
                attention_mask=model_mask,
                position_ids=prefill_pos_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True,
            )
            past_key_values = prefill_outputs.past_key_values

        num_transfer_tokens = self.get_bd3lm_num_transfer_tokens(block_size, steps)

        for block_idx in range(prefill_blocks, num_blocks):
            block_start = block_idx * block_size
            block_end = block_start + block_size

            cur_block_embeds = x_embeds[:, block_start:block_end].clone()
            cur_block_ids = x_ids[:, block_start:block_end]

            cur_mask = block_diffusion_mask[:, :, block_start:block_end, :block_end]
            cur_pos_ids = position_ids[:, block_start:block_end]
            model_mask = {"full_attention": cur_mask, "sliding_attention": cur_mask}

            for step in range(steps + 1):
                is_mask = torch.all(torch.abs(cur_block_embeds - mask_embed) < 1e-5, dim=-1)
                if not is_mask.any():
                    _ = self.model(
                        inputs_embeds=cur_block_embeds,
                        attention_mask=model_mask,
                        position_ids=cur_pos_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        store_kv=True,
                    )
                    break

                outputs = self.model(
                    inputs_embeds=cur_block_embeds,
                    attention_mask=model_mask,
                    position_ids=cur_pos_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False,
                )
                logits = self.lm_head(outputs[0]).float()

                top_k = kwargs.get("top_k", 0)
                top_p = kwargs.get("top_p", 1.0)
                x0, x0_p = self._sample_with_temperature_topk_topp(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                remasking_strategy = kwargs.get("remasking_strategy", "low_confidence_static")
                num_to_transfer = num_transfer_tokens[step].item()

                transfer_mask = torch.zeros_like(x0, dtype=torch.bool, device=device)
                if remasking_strategy == "low_confidence_static":
                    confidence = torch.where(is_mask, x0_p, -torch.inf)
                    for j in range(confidence.shape[0]):
                        num_masks = is_mask[j].sum().item()
                        k = min(num_to_transfer, num_masks)
                        if k > 0 and not torch.all(torch.isinf(confidence[j])):
                            _, idx = torch.topk(confidence[j], k)
                            transfer_mask[j, idx] = True
                elif remasking_strategy == "low_confidence_dynamic":
                    confidence_threshold = kwargs.get("confidence_threshold", 0.85)
                    confidence = torch.where(is_mask, x0_p, -torch.inf)
                    for j in range(confidence.shape[0]):
                        high_conf_mask = confidence[j] > confidence_threshold
                        num_high_confidence = high_conf_mask.sum().item()
                        if num_high_confidence >= num_to_transfer:
                            transfer_mask[j] = high_conf_mask
                        else:
                            num_masks = is_mask[j].sum().item()
                            k = min(num_to_transfer, num_masks)
                            if k > 0:
                                _, idx = torch.topk(confidence[j], k)
                                transfer_mask[j, idx] = True
                else:
                    raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")

                cur_block_ids = torch.where(transfer_mask, x0, cur_block_ids)
                x0_embeds = self.get_input_embeddings()(x0)
                cur_block_embeds = torch.where(transfer_mask.unsqueeze(-1), x0_embeds, cur_block_embeds)

            x_embeds[:, block_start:block_end] = cur_block_embeds
            x_ids[:, block_start:block_end] = cur_block_ids

            if block_end > prompt_len:
                gen_start_in_block = max(prompt_len, block_start)
                gen_ids_check = x_ids[:, gen_start_in_block:block_end]
                eos_token_id = 151645
                if eos_token_id in gen_ids_check:
                    break

        return x_ids[:, prompt_len:prompt_len + gen_length]

    @staticmethod
    def _top_k_logits(logits, k):
        if k <= 0:
            return logits
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    @staticmethod
    def _top_p_logits(logits, p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool), -1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask_indices, float("-inf"))
        return logits

    def _sample_with_temperature_topk_topp(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        """Sample with temperature, top-k, and top-p."""
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits_2d = logits.reshape(-1, vocab_size)

        if temperature == 0:
            token = torch.argmax(logits_2d, dim=-1, keepdim=True)
            probs_original = F.softmax(logits_2d, dim=-1)
            token_prob = torch.gather(probs_original, -1, token)
        else:
            logits_modified = logits_2d.clone()
            if temperature != 1.0:
                logits_modified = logits_modified / temperature
            if top_k > 0:
                logits_modified = self._top_k_logits(logits_modified, top_k)
            if top_p < 1.0:
                logits_modified = self._top_p_logits(logits_modified, top_p)
            probs_modified = F.softmax(logits_modified, dim=-1)
            token = torch.multinomial(probs_modified, num_samples=1)
            token_prob = torch.gather(probs_modified, -1, token)

        return token.view(*orig_shape), token_prob.view(*orig_shape)

    def get_bd3lm_num_transfer_tokens(self, block_length, steps):
        if steps == 0:
            return torch.zeros(0, dtype=torch.int64)
        base = block_length // steps
        remainder = block_length % steps
        num_transfer_tokens = torch.zeros(steps + 1, dtype=torch.int64) + base
        num_transfer_tokens[:remainder] += 1
        return num_transfer_tokens


class DiffusionVLQwen3VLMoeMultiModalModel(LlavaMetaModel, DiffusionVLQwen3VLMoeTextModel):
    """Multimodal model combining vision and text."""

    config_class = DiffusionVLQwen3VLMoeConfig

    def __init__(self, config):
        super(DiffusionVLQwen3VLMoeMultiModalModel, self).__init__(config)


class DiffusionVLQwen3VLMoeForCausalLM(DiffusionVLQwen3VLMoeForCausalLM_Base, LlavaMetaForCausalLM):
    """Final multimodal CausalLM model."""

    config_class = DiffusionVLQwen3VLMoeConfig

    def __init__(self, config):
        super(DiffusionVLQwen3VLMoeForCausalLM, self).__init__(config)
        self.model = DiffusionVLQwen3VLMoeMultiModalModel(config)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        image_grid_thws: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = None,
        return_dict: Optional[bool] = None,
        dsa_group_sizes: Optional[torch.Tensor] = None,
        dsa_answer_idx: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                modalities=modalities,
                image_sizes=image_sizes,
                image_grid_thws=image_grid_thws,
            )

        return super(DiffusionVLQwen3VLMoeForCausalLM, self).forward(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            dsa_group_sizes=dsa_group_sizes,
            dsa_answer_idx=dsa_answer_idx,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        image_grid_thws: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if modalities is None:
            modalities = ["image"]

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if images is not None:
            (_, _, _, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                modalities=modalities,
                image_sizes=image_sizes,
                image_grid_thws=image_grid_thws,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        kwargs.pop("input_ids", None)
        return self.generate_with_bd3lm(inputs_embeds=inputs_embeds, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        print(f">>> Loading pre-converted DiffusionVL-Qwen3VL-MoE model from: {pretrained_model_name_or_path}")

        model = super(DiffusionVLQwen3VLMoeForCausalLM, cls).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )
        model.config.tie_word_embeddings = False

        vision_config_path = os.path.join(pretrained_model_name_or_path, "vision_config.json")
        if os.path.exists(vision_config_path):
            print(">>> Loading pre-converted visual components from .safetensors files...")
            with open(vision_config_path, "r") as f:
                vision_config_dict = json.load(f)
            model.config.vision_config = PretrainedConfig.from_dict(vision_config_dict)
            model.config.vision_tower_state_dict = load_file(
                os.path.join(pretrained_model_name_or_path, "vision_tower.safetensors"),
                device="cpu",
            )
            model.config.projector_state_dict = load_file(
                os.path.join(pretrained_model_name_or_path, "projector.safetensors"),
                device="cpu",
            )

        print(">>> DiffusionVL-Qwen3VL-MoE model loaded successfully.")
        return model


AutoConfig.register("diffusionvl_qwen3vl_moe", DiffusionVLQwen3VLMoeConfig)
AutoModelForCausalLM.register(DiffusionVLQwen3VLMoeConfig, DiffusionVLQwen3VLMoeForCausalLM)
