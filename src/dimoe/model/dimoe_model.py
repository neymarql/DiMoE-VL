from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from dimoe.model.dmar import DMARRouter
from dimoe.utils.logging import setup_logger


class VisionTower(nn.Module):
    def __init__(
        self,
        model_name: str,
        hidden_size: int,
        device: torch.device,
        allow_dummy: bool = True,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.logger = setup_logger()
        self.model_name = model_name
        self.device = device
        self.allow_dummy = allow_dummy
        self.is_dummy = False

        try:
            try:
                self.processor = AutoImageProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                )
            except Exception:
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                )
            loaded_model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            # Some checkpoints (e.g. SigLIP) expose a multi-branch wrapper model.
            # Use the pure vision branch if present so forward only needs pixel values.
            self.model = loaded_model.vision_model if hasattr(loaded_model, "vision_model") else loaded_model
            self.model = self.model.to(device)
            with torch.no_grad():
                out = self.model(**self.processor(images=Image.new("RGB", (32, 32), color=(0, 0, 0)), return_tensors="pt").to(device))
                vdim = out.last_hidden_state.shape[-1]
            self.proj = nn.Linear(vdim, hidden_size)
        except Exception as exc:
            if not allow_dummy:
                raise
            self.logger.warning("vision tower fallback to dummy: %s", exc)
            self.processor = None
            self.model = None
            self.is_dummy = True
            self.proj = nn.Linear(hidden_size, hidden_size)

    def prepare_inputs(self, image_paths: List[str]) -> Optional[Dict[str, torch.Tensor]]:
        if self.is_dummy:
            return None

        imgs = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), color=(0, 0, 0))
            imgs.append(img)
        proc = self.processor(images=imgs, return_tensors="pt")
        return {k: v for k, v in proc.items()}

    def encode_vision_inputs(
        self,
        vision_inputs: Optional[Dict[str, torch.Tensor]],
        dtype: torch.dtype,
        batch_size: int,
    ) -> torch.Tensor:
        if self.is_dummy:
            x = torch.zeros((batch_size, self.proj.in_features), device=self.proj.weight.device, dtype=dtype)
            return self.proj(x)

        if vision_inputs is None:
            return torch.zeros((batch_size, self.proj.out_features), device=self.proj.weight.device, dtype=dtype)

        inputs = {k: v.to(self.device) for k, v in vision_inputs.items()}
        out = self.model(**inputs)
        feat = out.last_hidden_state.mean(dim=1)
        return self.proj(feat).to(dtype=dtype)

    def encode_paths(self, image_paths: List[str], dtype: torch.dtype) -> torch.Tensor:
        vision_inputs = self.prepare_inputs(image_paths)
        return self.encode_vision_inputs(vision_inputs=vision_inputs, dtype=dtype, batch_size=len(image_paths))


class DimoeModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        vision_tower: str,
        num_experts: int,
        num_t_buckets: int,
        num_token_types: int,
        image_token: str,
        mask_token: str,
        device: torch.device,
        allow_dummy_vision: bool = True,
        backbone_trust_remote_code: bool = True,
        vision_trust_remote_code: bool = True,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.logger = setup_logger()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone,
            trust_remote_code=backbone_trust_remote_code,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        special_to_add = []
        for tok in [image_token, mask_token]:
            if tok not in self.tokenizer.get_vocab():
                special_to_add.append(tok)
        if special_to_add:
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_to_add})

        backbone_cfg = AutoConfig.from_pretrained(
            backbone,
            trust_remote_code=backbone_trust_remote_code,
            local_files_only=local_files_only,
        )
        # Compatibility for LLaDA-MoE remote code on newer transformers:
        # make rope_scaling explicit so rope_type does not fall back to removed "default".
        rope_scaling = getattr(backbone_cfg, "rope_scaling", None)
        if rope_scaling is None:
            backbone_cfg.rope_scaling = {"rope_type": "linear", "factor": 1.0}
        elif isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rope_type == "default":
                rope_scaling["rope_type"] = "linear"
                rope_scaling["factor"] = float(rope_scaling.get("factor", 1.0))
                if "type" in rope_scaling:
                    rope_scaling["type"] = "linear"
                backbone_cfg.rope_scaling = rope_scaling

        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone,
            config=backbone_cfg,
            trust_remote_code=backbone_trust_remote_code,
            local_files_only=local_files_only,
        )
        self.backbone.resize_token_embeddings(len(self.tokenizer))

        hidden = self.backbone.config.hidden_size
        self.vision = VisionTower(
            vision_tower,
            hidden_size=hidden,
            device=device,
            allow_dummy=allow_dummy_vision,
            trust_remote_code=vision_trust_remote_code,
            local_files_only=local_files_only,
        )
        self.dmar_router = DMARRouter(hidden, num_experts=num_experts, num_t_buckets=num_t_buckets, num_token_types=num_token_types)

        self.image_token = image_token
        self.mask_token = mask_token
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(mask_token)

        self.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        t_bucket: torch.Tensor,
        token_type_ids: torch.Tensor,
        vision_inputs: Optional[Dict[str, torch.Tensor]] = None,
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_diffusion(
            input_ids=input_ids,
            attention_mask=attention_mask,
            t_bucket=t_bucket,
            token_type_ids=token_type_ids,
            vision_inputs=vision_inputs,
            image_paths=image_paths,
        )

    def build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        image_paths: Optional[List[str]] = None,
        vision_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        emb = self.backbone.get_input_embeddings()(input_ids)
        if vision_inputs is None and not image_paths:
            return emb

        if vision_inputs is not None:
            vision_feats = self.vision.encode_vision_inputs(
                vision_inputs=vision_inputs,
                dtype=emb.dtype,
                batch_size=input_ids.shape[0],
            )
        else:
            vision_feats = self.vision.encode_paths(image_paths or [], dtype=emb.dtype)
        # Replace first image token occurrence per sample by projected vision vector.
        for i in range(input_ids.shape[0]):
            pos = (input_ids[i] == self.image_token_id).nonzero(as_tuple=False)
            if pos.numel() == 0:
                continue
            first = int(pos[0].item())
            emb[i, first, :] = vision_feats[i]
        return emb

    def forward_diffusion(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        t_bucket: torch.Tensor,
        token_type_ids: torch.Tensor,
        vision_inputs: Optional[Dict[str, torch.Tensor]] = None,
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        t_bucket = t_bucket.to(self.device)
        token_type_ids = token_type_ids.to(self.device)

        embeds = self.build_inputs_embeds(
            input_ids,
            image_paths=image_paths or [],
            vision_inputs=vision_inputs,
        )

        outputs = self.backbone(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits
        hidden = outputs.hidden_states[-1]

        gate_probs = None
        if hasattr(outputs, "router_logits") and outputs.router_logits is not None:
            # some MoE models expose layer-wise router logits
            rt = outputs.router_logits
            if isinstance(rt, (list, tuple)) and rt:
                last = rt[-1]
                if last.dim() == 3:
                    gate_probs = torch.softmax(last, dim=-1)
                elif last.dim() == 2:
                    # [B*L, E]
                    b, l = hidden.shape[:2]
                    gate_probs = torch.softmax(last, dim=-1).view(b, l, -1)
        dmar_probs = self.dmar_router(hidden, t_bucket=t_bucket, token_type=token_type_ids)
        if gate_probs is None:
            gate_probs = dmar_probs
        else:
            # Use blended routing probabilities for DMAR objectives/diagnostics
            # while keeping backbone expert dispatch unchanged.
            gate_probs = 0.5 * (gate_probs + dmar_probs)

        return {
            "logits": logits,
            "hidden_states": hidden,
            "gate_probs": gate_probs,
        }

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def encode_text(self, text: str, max_length: int = 4096) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]

    def load_checkpoint(self, path: str | Path, strict: bool = False) -> Dict:
        ckpt = torch.load(str(path), map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        keys = self.load_state_dict(state, strict=strict)
        missing = keys.missing_keys
        unexpected = keys.unexpected_keys
        self.logger.info(
            "loaded checkpoint %s (missing=%d unexpected=%d)",
            path,
            len(missing),
            len(unexpected),
        )
        return ckpt if isinstance(ckpt, dict) else {"raw_checkpoint": True}
