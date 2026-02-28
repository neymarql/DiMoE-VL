from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dimoe.model.dmar import DMARLossOutput, dmar_losses
from dimoe.model.losses import masked_ce_loss
from dimoe.model.masking import TimestepSampler, apply_absorbing_mask, build_complementary_masks
from dimoe.train.dataset import DimoeCollator, DimoeJsonlDataset
from dimoe.train.stage import stage_curriculum
from dimoe.utils.config import load_yaml
from dimoe.utils.io import write_json
from dimoe.utils.logging import JsonlWriter, setup_logger


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    if _is_dist():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def _world() -> int:
    if _is_dist():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def _local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    return 0


def _rank0() -> bool:
    return _rank() == 0


def _unwrap_model(model):
    if isinstance(model, DDP):
        return model.module
    if hasattr(model, "module"):
        return model.module
    return model


def maybe_init_dist() -> None:
    if _world() > 1 and not _is_dist():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout_s = int(os.environ.get("DIST_TIMEOUT_SEC", "7200"))
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_s))


def _save_ckpt(path: Path, model, optim: AdamW, state: TrainState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _unwrap_model(model)
    ckpt = {
        "model": raw.state_dict(),
        "optimizer": optim.state_dict(),
        "step": state.step,
        "epoch": state.epoch,
    }
    torch.save(ckpt, path)


def _save_handoff_ckpt(path: Path, model, state: TrainState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _unwrap_model(model)
    ckpt = {
        "model": raw.state_dict(),
        "step": state.step,
        "epoch": state.epoch,
    }
    torch.save(ckpt, path)


def _load_ckpt(path: Path, model, optim: AdamW, device: torch.device) -> TrainState:
    raw = _unwrap_model(model)
    ckpt = torch.load(path, map_location="cpu")
    raw.load_state_dict(ckpt["model"], strict=False)
    if "optimizer" in ckpt:
        optim.load_state_dict(ckpt["optimizer"])
        for st in optim.state.values():
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.to(device)
    return TrainState(step=int(ckpt.get("step", 0)), epoch=int(ckpt.get("epoch", 0)))


def _latest_ckpt(ckpt_dir: Path) -> Optional[Path]:
    cks = sorted(ckpt_dir.glob("global_step_*.pt"))
    return cks[-1] if cks else None


def _latest_ds_tag(ckpt_dir: Path) -> Optional[str]:
    tags = sorted([p.name for p in ckpt_dir.glob("global_step_*") if p.is_dir()])
    return tags[-1] if tags else None


def _bucketize_t(t: torch.Tensor, n_buckets: int) -> torch.Tensor:
    tb = torch.clamp((t * n_buckets).long(), 0, n_buckets - 1)
    return tb


def _expert_grad_norm(model) -> float:
    raw = _unwrap_model(model)
    s = 0.0
    for n, p in raw.named_parameters():
        if p.grad is None:
            continue
        if "expert" in n.lower() or "router" in n.lower():
            s += float(p.grad.detach().norm().item())
    return s


def _vision_grad_norm(model) -> float:
    raw = _unwrap_model(model)
    s = 0.0
    for n, p in raw.named_parameters():
        if p.grad is None:
            continue
        if "vision" in n.lower() or "proj" in n.lower():
            s += float(p.grad.detach().norm().item())
    return s


def _apply_train_policy(model, cfg: Dict) -> None:
    train_cfg = cfg.get("train", {})
    if bool(train_cfg.get("gradient_checkpointing", True)) and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()
    if hasattr(model.backbone, "config") and hasattr(model.backbone.config, "use_cache"):
        model.backbone.config.use_cache = False

    if bool(train_cfg.get("freeze_backbone", False)):
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    if bool(train_cfg.get("freeze_vision", False)):
        for p in model.vision.parameters():
            p.requires_grad_(False)

    train_router_only = bool(train_cfg.get("train_router_only", False))
    if train_router_only:
        for n, p in model.named_parameters():
            keep = "dmar_router" in n or "vision.proj" in n
            p.requires_grad_(keep)


def _sync_mean_scalar(x: torch.Tensor) -> torch.Tensor:
    if _is_dist():
        y = x.detach().clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        y = y / _world()
        return y
    return x.detach()


def run_training(cfg: Dict):
    logger = setup_logger()
    maybe_init_dist()

    local_rank = _local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    stage = cfg["stage"]
    cur = stage_curriculum(stage)
    for k, v in cfg.get("curriculum", {}).items():
        if hasattr(cur, k):
            setattr(cur, k, v)

    out_dir = Path(cfg["output_dir"]) / stage
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_writer = JsonlWriter(log_dir / "train_metrics.jsonl")
    route_writer = JsonlWriter(log_dir / "routing_stats.jsonl")
    grad_writer = JsonlWriter(log_dir / "gradient_stats.jsonl")
    heartbeat_writer = JsonlWriter(log_dir / "heartbeat.jsonl")

    from dimoe.model.dimoe_model import DimoeModel

    raw_model = DimoeModel(
        backbone=cfg["model"]["backbone"],
        vision_tower=cfg["model"]["vision_tower"],
        num_experts=int(cfg["model"]["num_experts"]),
        num_t_buckets=int(cfg["model"]["num_t_buckets"]),
        num_token_types=int(cfg["model"]["num_token_types"]),
        image_token=cfg["model"]["image_token"],
        mask_token=cfg["model"]["mask_token"],
        device=device,
        allow_dummy_vision=bool(cfg["model"].get("allow_dummy_vision", True)),
        backbone_trust_remote_code=bool(cfg["model"].get("backbone_trust_remote_code", True)),
        vision_trust_remote_code=bool(cfg["model"].get("vision_trust_remote_code", True)),
        local_files_only=bool(cfg["model"].get("local_files_only", False)),
    )
    init_ckpt = str(cfg["model"].get("init_checkpoint", ""))
    if init_ckpt:
        raw_model.load_checkpoint(init_ckpt, strict=False)
    _apply_train_policy(raw_model, cfg)

    train_jsonl = Path(cfg["data"]["train_jsonl"])
    if not train_jsonl.exists():
        raise FileNotFoundError(f"train jsonl not found: {train_jsonl}")

    ds = DimoeJsonlDataset(
        jsonl_path=str(train_jsonl),
        tokenizer=raw_model.tokenizer,
        max_length=int(cfg["train"]["max_length"]),
        image_token=cfg["model"]["image_token"],
    )
    if len(ds) == 0:
        raise ValueError(f"empty training dataset: {train_jsonl}")

    sampler = DistributedSampler(ds, num_replicas=_world(), rank=_rank(), shuffle=True) if _world() > 1 else None
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["micro_batch_size"]),
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=DimoeCollator(
            pad_token_id=raw_model.tokenizer.pad_token_id or raw_model.tokenizer.eos_token_id,
            fixed_length=(
                int(cfg["train"]["max_length"]) if bool(cfg["train"].get("pad_to_max_length", True)) else None
            ),
        ),
        num_workers=int(cfg["train"].get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(cfg["train"].get("drop_last", False)),
    )

    train_cfg = cfg.get("train", {})
    engine_name = str(train_cfg.get("engine", "ddp")).lower()
    use_deepspeed = engine_name == "deepspeed"

    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"].get("weight_decay", 0.01))
    trainable = [p for p in raw_model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters after applying train policy")

    model = raw_model
    optim: Optional[AdamW] = None
    if use_deepspeed:
        try:
            import deepspeed  # type: ignore
        except Exception as exc:
            raise RuntimeError("engine=deepspeed but deepspeed is not available") from exc

        ds_cfg_path = str(train_cfg.get("deepspeed_config", "")).strip()
        if not ds_cfg_path:
            raise ValueError("engine=deepspeed requires train.deepspeed_config")
        ds_cfg_file = Path(ds_cfg_path)
        if not ds_cfg_file.exists():
            raise FileNotFoundError(f"deepspeed config not found: {ds_cfg_file}")

        ds_cfg = json.loads(ds_cfg_file.read_text(encoding="utf-8"))
        # Keep DeepSpeed GAS at 1 and do grad accumulation explicitly in this loop.
        ds_cfg["gradient_accumulation_steps"] = 1
        ds_cfg["train_micro_batch_size_per_gpu"] = int(train_cfg.get("micro_batch_size", 1))
        ds_cfg.setdefault("gradient_clipping", float(train_cfg.get("clip_grad", 1.0)))
        ds_cfg.setdefault(
            "optimizer",
            {
                "type": "AdamW",
                "params": {
                    "lr": lr,
                    "betas": [0.9, 0.95],
                    "eps": 1.0e-8,
                    "weight_decay": wd,
                    "torch_adam": True,
                },
            },
        )
        if isinstance(ds_cfg.get("optimizer"), dict):
            ds_cfg["optimizer"].setdefault("params", {})
            ds_cfg["optimizer"]["params"]["lr"] = lr
            ds_cfg["optimizer"]["params"]["weight_decay"] = wd
            ds_cfg["optimizer"]["params"].setdefault("torch_adam", True)
        if bool(train_cfg.get("bf16", True)):
            ds_cfg.setdefault("bf16", {})
            ds_cfg["bf16"]["enabled"] = True

        model, optim, _, _ = deepspeed.initialize(
            model=raw_model,
            model_parameters=trainable,
            config=ds_cfg,
        )
        logger.info("initialized deepspeed engine (world=%d)", _world())
        try:
            logger.info(
                "deepspeed effective gas=%s micro_batch=%s",
                model.gradient_accumulation_steps(),
                model.train_micro_batch_size_per_gpu(),
            )
        except Exception:
            pass
    else:
        if _world() > 1:
            model = DDP(
                raw_model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=bool(cfg["train"].get("find_unused_parameters", False)),
            )
        optim = AdamW(trainable, lr=lr, weight_decay=wd)

    state = TrainState()
    if cfg["train"].get("resume", True):
        if use_deepspeed:
            latest_tag = _latest_ds_tag(ckpt_dir)
            if latest_tag is not None:
                load_path, client_state = model.load_checkpoint(
                    str(ckpt_dir),
                    tag=latest_tag,
                    load_module_only=False,
                    load_optimizer_states=True,
                    load_lr_scheduler_states=False,
                )
                if load_path:
                    state = TrainState(
                        step=int((client_state or {}).get("step", 0)),
                        epoch=int((client_state or {}).get("epoch", 0)),
                    )
                    logger.info("resumed deepspeed from %s step=%d", load_path, state.step)
        else:
            latest = _latest_ckpt(ckpt_dir)
            if latest is not None:
                assert optim is not None
                state = _load_ckpt(latest, model, optim, device=device)
                logger.info("resumed from %s step=%d", latest, state.step)

    max_steps = int(cfg["train"]["max_steps"])
    grad_accum = int(cfg["train"].get("grad_accum", 1))
    save_every = int(cfg["train"].get("save_every", 200))
    log_every = int(cfg["train"].get("log_every", 10))

    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if bool(cfg["train"].get("bf16", True)) else torch.float16

    ts1 = TimestepSampler(mode=cur.t_mode_1, late_bias=cur.late_bias)
    ts2 = TimestepSampler(mode=cur.t_mode_2, late_bias=cur.late_bias)

    model.train()
    if optim is not None:
        optim.zero_grad(set_to_none=True)

    pbar = tqdm(total=max_steps, disable=not _rank0(), desc=f"train:{stage}")
    pbar.update(min(state.step, max_steps))

    micro_step = 0
    run_start_time = time.time()
    last_opt_step_time = run_start_time
    heartbeat_every_micro = bool(train_cfg.get("heartbeat_every_micro_step", False))
    heartbeat_every_step = bool(train_cfg.get("heartbeat_every_step", True))
    while state.step < max_steps:
        progressed = False
        if sampler is not None:
            sampler.set_epoch(state.epoch)

        for batch in dl:
            if state.step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            prompt_mask = batch["prompt_mask"].to(device, non_blocking=True)
            token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            image_paths = batch["image_paths"]
            vision_inputs = raw_model.vision.prepare_inputs(image_paths)

            comp = build_complementary_masks(
                labels=labels,
                prompt_mask=prompt_mask,
                include_prompt_ratio=cur.include_prompt_ratio,
                timestep_sampler_1=ts1,
                timestep_sampler_2=ts2,
                force_answer_coverage=True,
            )

            x1 = apply_absorbing_mask(input_ids, comp.mask1, mask_token_id=raw_model.mask_token_id)
            x2 = apply_absorbing_mask(input_ids, comp.mask2, mask_token_id=raw_model.mask_token_id)

            tb1 = _bucketize_t(comp.t1, int(cfg["model"]["num_t_buckets"]))[:, None].repeat(1, input_ids.shape[1])
            tb2 = _bucketize_t(comp.t2, int(cfg["model"]["num_t_buckets"]))[:, None].repeat(1, input_ids.shape[1])

            tt1 = token_type_ids.clone()
            tt2 = token_type_ids.clone()
            tt1[comp.mask1] = 3
            tt2[comp.mask2] = 3

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out1 = model(
                    input_ids=x1,
                    attention_mask=attention_mask,
                    t_bucket=tb1,
                    token_type_ids=tt1,
                    vision_inputs=vision_inputs,
                )
                out2 = model(
                    input_ids=x2,
                    attention_mask=attention_mask,
                    t_bucket=tb2,
                    token_type_ids=tt2,
                    vision_inputs=vision_inputs,
                )

                l1 = masked_ce_loss(out1["logits"], input_ids, comp.mask1)
                l2 = masked_ce_loss(out2["logits"], input_ids, comp.mask2)

                prefix_mask = token_type_ids.le(1)
                use_dmar_objective = any(
                    float(x) > 0.0
                    for x in (cur.lambda_prior, cur.lambda_cons, cur.lambda_bal, cur.lambda_entropy)
                )
                if use_dmar_objective:
                    dmar = dmar_losses(
                        gate_probs=out1["gate_probs"],
                        gate_probs_other=out2["gate_probs"],
                        prefix_mask=prefix_mask,
                        t_bucket=tb1,
                        token_type=tt1,
                        prior_table=raw_model.dmar_router.prior,
                        lambda_prior=cur.lambda_prior,
                        lambda_consistency=cur.lambda_cons,
                        lambda_balance=cur.lambda_bal,
                        lambda_entropy=cur.lambda_entropy,
                    )
                else:
                    z = l1.loss.new_zeros(())
                    dmar = DMARLossOutput(
                        prior_loss=z,
                        consistency_loss=z,
                        balance_loss=z,
                        entropy_loss=z,
                        stats={
                            "dmar_prior": 0.0,
                            "dmar_consistency": 0.0,
                            "dmar_balance": 0.0,
                            "dmar_entropy": 0.0,
                        },
                    )

                total = l1.loss + l2.loss + dmar.prior_loss + dmar.consistency_loss + dmar.balance_loss + dmar.entropy_loss
                loss = total / grad_accum

            if not torch.isfinite(total.detach()).all():
                raise FloatingPointError(
                    f"non-finite total loss at step={state.step} "
                    f"(l1_tokens={l1.token_count}, l2_tokens={l2.token_count}, "
                    f"l1={float(l1.loss.detach().float().item())}, "
                    f"l2={float(l2.loss.detach().float().item())}, "
                    f"dmar_prior={float(dmar.prior_loss.detach().float().item())}, "
                    f"dmar_cons={float(dmar.consistency_loss.detach().float().item())}, "
                    f"dmar_bal={float(dmar.balance_loss.detach().float().item())}, "
                    f"dmar_ent={float(dmar.entropy_loss.detach().float().item())})"
                )
            total_scalar = float(total.detach().float().item())
            loss1_scalar = float(l1.loss.detach().float().item())
            loss2_scalar = float(l2.loss.detach().float().item())

            if use_deepspeed:
                model.backward(loss)
                micro_step += 1
                if _rank0() and heartbeat_every_micro:
                    accum_idx = micro_step % grad_accum
                    if accum_idx == 0:
                        accum_idx = grad_accum
                    heartbeat_writer.write(
                        {
                            "kind": "micro",
                            "step": state.step,
                            "epoch": state.epoch,
                            "micro_step": micro_step,
                            "accum_idx": accum_idx,
                            "grad_accum": grad_accum,
                            "loss_total": total_scalar,
                            "loss_branch1": loss1_scalar,
                            "loss_branch2": loss2_scalar,
                            "tokens_branch1": l1.token_count,
                            "tokens_branch2": l2.token_count,
                            "elapsed_s": time.time() - run_start_time,
                        }
                    )
                    logger.info(
                        "heartbeat micro step=%d accum=%d/%d state_step=%d loss=%.4f",
                        micro_step,
                        accum_idx,
                        grad_accum,
                        state.step,
                        total_scalar,
                    )
                if micro_step % grad_accum != 0:
                    continue
                model.step()
            else:
                assert optim is not None
                loss.backward()
                micro_step += 1
                if _rank0() and heartbeat_every_micro:
                    accum_idx = micro_step % grad_accum
                    if accum_idx == 0:
                        accum_idx = grad_accum
                    heartbeat_writer.write(
                        {
                            "kind": "micro",
                            "step": state.step,
                            "epoch": state.epoch,
                            "micro_step": micro_step,
                            "accum_idx": accum_idx,
                            "grad_accum": grad_accum,
                            "loss_total": total_scalar,
                            "loss_branch1": loss1_scalar,
                            "loss_branch2": loss2_scalar,
                            "tokens_branch1": l1.token_count,
                            "tokens_branch2": l2.token_count,
                            "elapsed_s": time.time() - run_start_time,
                        }
                    )
                    logger.info(
                        "heartbeat micro step=%d accum=%d/%d state_step=%d loss=%.4f",
                        micro_step,
                        accum_idx,
                        grad_accum,
                        state.step,
                        total_scalar,
                    )

                if micro_step % grad_accum != 0:
                    continue

                torch.nn.utils.clip_grad_norm_(trainable, float(cfg["train"].get("clip_grad", 1.0)))
                optim.step()
                optim.zero_grad(set_to_none=True)

            should_log = (state.step % log_every) == 0
            total_mean = total_scalar
            loss1_mean = loss1_scalar
            loss2_mean = loss2_scalar
            now_time = time.time()
            opt_step_wall = now_time - last_opt_step_time
            last_opt_step_time = now_time
            if _rank0() and heartbeat_every_step:
                heartbeat_writer.write(
                    {
                        "kind": "step",
                        "step": state.step,
                        "epoch": state.epoch,
                        "micro_step": micro_step,
                        "grad_accum": grad_accum,
                        "loss_total": total_mean,
                        "loss_branch1": loss1_mean,
                        "loss_branch2": loss2_mean,
                        "tokens_branch1": l1.token_count,
                        "tokens_branch2": l2.token_count,
                        "step_wall_s": opt_step_wall,
                        "elapsed_s": now_time - run_start_time,
                    }
                )
                logger.info(
                    "heartbeat step=%d epoch=%d loss=%.4f step_wall=%.2fs elapsed=%.1fs",
                    state.step,
                    state.epoch,
                    total_mean,
                    opt_step_wall,
                    now_time - run_start_time,
                )

            if _rank0() and should_log:
                metric_writer.write(
                    {
                        "step": state.step,
                        "loss_total": float(total_mean),
                        "loss_branch1": float(loss1_mean),
                        "loss_branch2": float(loss2_mean),
                        "tokens_branch1": l1.token_count,
                        "tokens_branch2": l2.token_count,
                        **dmar.stats,
                    }
                )
                route_writer.write(
                    {
                        "step": state.step,
                        "gate_mean": out1["gate_probs"].detach().mean(dim=(0, 1)).cpu().tolist(),
                        "gate_top1": out1["gate_probs"].detach().argmax(dim=-1).float().mean().item(),
                    }
                )
                grad_writer.write(
                    {
                        "step": state.step,
                        "vision_grad_norm": _vision_grad_norm(model),
                        "expert_grad_norm": _expert_grad_norm(model),
                    }
                )

            state.step += 1
            progressed = True
            if _rank0():
                pbar.update(1)

            if state.step > 0 and state.step % save_every == 0:
                if use_deepspeed:
                    tag = f"global_step_{state.step:09d}"
                    model.save_checkpoint(str(ckpt_dir), tag=tag, client_state={"step": state.step, "epoch": state.epoch})
                    if _rank0():
                        _save_handoff_ckpt(ckpt_dir / f"{tag}.pt", model, state)
                elif _rank0():
                    assert optim is not None
                    _save_ckpt(ckpt_dir / f"global_step_{state.step:09d}.pt", model, optim, state)

            if state.step >= max_steps:
                break

        state.epoch += 1
        if not progressed:
            raise RuntimeError(
                "no optimizer step was made in this epoch; "
                "check dataset size, grad_accum settings, and deepspeed boundary behavior"
            )

    if _rank0():
        pbar.close()

    if use_deepspeed:
        tag = f"global_step_{state.step:09d}"
        model.save_checkpoint(str(ckpt_dir), tag=tag, client_state={"step": state.step, "epoch": state.epoch})
        if _rank0():
            _save_handoff_ckpt(ckpt_dir / f"{tag}.pt", model, state)
    elif _rank0():
        assert optim is not None
        _save_ckpt(ckpt_dir / f"global_step_{state.step:09d}.pt", model, optim, state)

    if _rank0():
        write_json(out_dir / "train_state.json", {"step": state.step, "epoch": state.epoch, "stage": stage})

    if _is_dist():
        dist.barrier()
        dist.destroy_process_group()


def add_parser(subparsers):
    p = subparsers.add_parser("run", help="Run stage training")
    p.add_argument("--stage", required=True, choices=["stage0_naive", "stage1_align", "stage2_full", "stage3_lownfe"])
    p.add_argument("--config", required=True, type=str)
    p.set_defaults(func=main)


def main(args: argparse.Namespace):
    cfg = load_yaml(args.config)
    cfg["stage"] = args.stage
    run_training(cfg)
