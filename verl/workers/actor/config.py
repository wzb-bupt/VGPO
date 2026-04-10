# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Actor config
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelConfig:
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    override_config: dict[str, Any] = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    freeze_vision_tower: bool = False

    def post_init(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

        if self.model_path is not None and os.path.exists(self.model_path):  # ray job uses absolute path
            self.model_path = os.path.abspath(self.model_path)

        if self.tokenizer_path is not None and os.path.exists(self.tokenizer_path):
            self.tokenizer_path = os.path.abspath(self.tokenizer_path)


@dataclass
class OptimConfig:
    lr: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    strategy: str = "adamw"
    lr_warmup_ratio: float = 0.0
    lr_warmup_steps: Optional[int] = None
    min_lr_ratio: Optional[float] = None
    warmup_style: str = "constant"
    # below are auto keys
    training_steps: int = field(default=-1, init=False)


@dataclass
class FSDPConfig:
    enable_full_shard: bool = True
    enable_cpu_offload: bool = False
    enable_rank0_init: bool = True
    use_orig_params: bool = False
    torch_dtype: Optional[str] = None
    fsdp_size: int = -1
    mp_param_dtype: str = "bf16"
    mp_reduce_dtype: str = "fp32"
    mp_buffer_dtype: str = "fp32"


@dataclass
class OffloadConfig:
    offload_params: bool = False
    offload_optimizer: bool = False


@dataclass
class ActorConfig:
    strategy: str = "fsdp"
    global_batch_size: int = 256
    """number of samples per minibatch for updating actor"""
    micro_batch_size_per_device_for_update: int = 4
    """number of samples per forward pass for updating actor"""
    micro_batch_size_per_device_for_experience: int = 16
    """number of samples per forward pass for computing log probs"""
    max_grad_norm: float = 1.0
    """number to clip grad norm"""
    clip_ratio_low: float = 0.2
    """clip ratio in PPO & DAPO"""
    clip_ratio_high: float = 0.3
    """clip ratio in PPO & DAPO"""
    clip_ratio_dual: float = 3.0
    """constant C in dual-clip PPO, clips when advantage < -C"""
    loss_avg_mode: str = "token"
    """loss average mode: `token`, `seq`"""
    loss_type: str = "default"
    """loss type: `default`, `gspo`, `cispo`"""
    ppo_epochs: int = 1
    """number of ppo epochs for each rollout batch"""
    padding_free: bool = True
    """use padding-free training"""
    dynamic_batching: bool = True
    """enable dynamic batching"""
    ulysses_size: int = 1
    """ulysses sequence parallel size"""
    use_torch_compile: bool = True
    """enable torch compile"""
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    # below are auto keys
    global_batch_size_per_device: int = field(default=-1, init=False)
    disable_kl: bool = field(default=False, init=False)
    use_kl_loss: bool = field(default=False, init=False)
    kl_penalty: str = field(default="kl", init=False)
    kl_coef: float = field(default=0.0, init=False)

    use_entropy_penalty: bool = False
    """use entropy penalty for training"""
    entropy_penalty_coef: float = 0.06
    """entropy penalty coefficient"""


    # Intra-trajectory Re-weighting (token-level visual attention reweighting within each trajectory)
    use_intra_trajectory_reweighting: bool = False
    """enable intra-trajectory reweighting: modulate token advantages by visual focus scores"""

    # Visual Attention Compensation (Eq.5: combat visual forgetting with progressive incentive)
    use_visual_compensation: bool = False
    """enable visual attention compensation to combat visual forgetting"""
    visual_compensation_type: str = "linear"
    """type of compensation schedule: 'linear', 'exponential', or 'step-function'"""
    visual_compensation_strength: float = 0.5
    """compensation intensity β (0.0 = no compensation, higher = stronger)"""
    visual_compensation_start_ratio: float = 0.0
    """start position ratio (0.0 = from beginning, 0.5 = from middle)"""
    visual_compensation_exponent: float = 2.0
    """exponent for exponential schedule (only used when type='exponential')"""
    visual_compensation_step_ratio: float = 0.5
    """step position ratio for step-function (only used when type='step-function', 0.0-1.0)"""

    # Gated Visual Compensation (Eq.6: only visual tokens get compensation)
    use_gated_visual_compensation: bool = False
    """enable gated visual compensation - only tokens above visual threshold get compensation"""
    visual_attention_threshold: float = 0.2
    """visual attention threshold κ for determining if a token is a 'visual token'"""
    gated_visual_compensation_start_ratio: float = 0.0
    """start position ratio for gated compensation - tokens from this position onwards are candidates for Top-κ% selection"""


    # Inter-trajectory Re-weighting (trajectory-level visual accumulation reweighting within rollout group)
    use_inter_trajectory_reweighting: bool = False
    """enable inter-trajectory reweighting: prioritize trajectories with superior visual accumulation"""
    inter_reweighting_gamma_late: float = 0.5
    """weight for late-stage visual attention score (score_late) in trajectory scoring"""


@dataclass
class RefConfig:
    strategy: str = "fsdp"
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    # below are auto keys
    micro_batch_size_per_device_for_experience: int = field(default=-1, init=False)
    padding_free: bool = field(default=False, init=False)
    dynamic_batching: bool = field(default=False, init=False)
    ulysses_size: int = field(default=1, init=False)
    use_torch_compile: bool = field(default=True, init=False)
