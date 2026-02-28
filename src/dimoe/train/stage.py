from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumCfg:
    name: str
    t_mode_1: str
    t_mode_2: str
    late_bias: float
    include_prompt_ratio: float
    lambda_bal: float
    lambda_prior: float
    lambda_cons: float
    lambda_entropy: float


def stage_curriculum(stage: str) -> CurriculumCfg:
    stage = stage.lower()
    if stage == "stage0_naive":
        return CurriculumCfg(
            name=stage,
            t_mode_1="uniform",
            t_mode_2="uniform",
            late_bias=0.0,
            include_prompt_ratio=0.0,
            lambda_bal=0.0,
            lambda_prior=0.0,
            lambda_cons=0.0,
            lambda_entropy=0.0,
        )
    if stage == "stage1_align":
        return CurriculumCfg(
            name=stage,
            t_mode_1="late_bias",
            t_mode_2="late_bias",
            late_bias=1.5,
            include_prompt_ratio=0.0,
            lambda_bal=0.05,
            lambda_prior=0.03,
            lambda_cons=0.05,
            lambda_entropy=0.01,
        )
    if stage == "stage2_full":
        return CurriculumCfg(
            name=stage,
            t_mode_1="uniform",
            t_mode_2="uniform",
            late_bias=0.0,
            include_prompt_ratio=0.15,
            lambda_bal=0.02,
            lambda_prior=0.02,
            lambda_cons=0.03,
            lambda_entropy=0.005,
        )
    if stage == "stage3_lownfe":
        return CurriculumCfg(
            name=stage,
            t_mode_1="uniform",
            t_mode_2="late_bias",
            late_bias=0.8,
            include_prompt_ratio=0.20,
            lambda_bal=0.01,
            lambda_prior=0.01,
            lambda_cons=0.02,
            lambda_entropy=0.0,
        )
    raise ValueError(f"unknown stage: {stage}")
