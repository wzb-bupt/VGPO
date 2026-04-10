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
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import (
    average_loss,
    compute_kl,
    compute_policy_loss,
    compute_trajectory_visual_weighting as compute_inter_trajectory_reweighting,
)
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

        if self.rank == 0:
            _active = []
            if getattr(config, 'use_visual_compensation', False):
                _active.append(f"VAC(β={getattr(config, 'visual_compensation_strength', 0.3)})")
            if getattr(config, 'use_gated_visual_compensation', False):
                _active.append(f"GatedVAC(γ={getattr(config, 'gated_visual_compensation_start_ratio', 0.5)}, κ={getattr(config, 'visual_attention_threshold', 0.2)})")
            if getattr(config, 'use_intra_trajectory_reweighting', False):
                _active.append("IntraReweight")
            if getattr(config, 'use_inter_trajectory_reweighting', False):
                _active.append("InterReweight")
            if _active:
                print(f"[VGPO Actor] Active modules: {', '.join(_active)}")
            else:
                print("[VGPO Actor] No VGPO modules enabled, running standard PPO actor.")

    def _compute_visual_attention_weights(
        self, 
        hidden_states: torch.Tensor, 
        input_ids: torch.Tensor, 
        response_length: int,
        prompt_length: int
    # ) -> Optional[torch.Tensor]:
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Compute visual attention weights based on hidden states similarity.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim) - last layer hidden states
            input_ids: (batch_size, seq_len) - input token ids
            response_length: length of response tokens
            prompt_length: length of prompt tokens
            
        Returns:
            tuple of (visual_attention_weights, mean_similarity) or None:
                - visual_attention_weights: (batch_size, response_length) - normalized weights
                - mean_similarity: scalar tensor - mean similarity before normalization
        """
        # if self.rank == 0: print("[INFO]: Using _compute_visual_attention_weights")
        
        # Get image_token_id from model config
        image_token_id = None
        if hasattr(self.actor_module, 'config') and hasattr(self.actor_module.config, 'image_token_id'):
            image_token_id = self.actor_module.config.image_token_id
        elif hasattr(self.actor_module, 'module') and hasattr(self.actor_module.module, 'config'):
            if hasattr(self.actor_module.module.config, 'image_token_id'):
                image_token_id = self.actor_module.module.config.image_token_id
        
        if image_token_id is None:
            if self.rank == 0:
                print("WARNING: image_token_id not found in model config. Visual attention weights will be None.")
            return None
        
        batch_size = hidden_states.size(0)
        visual_attention_weights = []
        all_similarities = []  # Store all similarities before normalization
        
        for batch_idx in range(batch_size):
            # Find visual token positions in the prompt
            visual_token_mask = (input_ids[batch_idx] == image_token_id)
            visual_positions = visual_token_mask.nonzero(as_tuple=True)[0]
            
            if len(visual_positions) == 0:
                # No visual tokens, return zero weights
                visual_attention_weights.append([0.0] * response_length)
                continue
            
            # Get visual tokens' hidden states and compute mean
            visual_hidden = hidden_states[batch_idx, visual_positions, :]  # (num_visual, hidden_dim)
            visual_hidden_mean = visual_hidden.mean(dim=0)  # (hidden_dim,)
            
            # Compute similarity for each generated token
            gen_weights = []
            for gen_idx in range(response_length):
                gen_pos = prompt_length + gen_idx
                if gen_pos >= hidden_states.size(1):
                    gen_weights.append(0.0)
                    continue
                    
                gen_hidden = hidden_states[batch_idx, gen_pos, :]  # (hidden_dim,)
                
                # Compute cosine similarity
                similarity = F.cosine_similarity(
                    gen_hidden.unsqueeze(0), 
                    visual_hidden_mean.unsqueeze(0), 
                    dim=1
                )
                # gen_weights.append(similarity.item())
                similarity_val = similarity.item()
                gen_weights.append(similarity_val)
                all_similarities.append(similarity_val)
            
            visual_attention_weights.append(gen_weights)
        
        visual_attention_weights = torch.tensor(
            visual_attention_weights, 
            device=hidden_states.device, 
            dtype=hidden_states.dtype
        )

        # Compute mean similarity before normalization
        if len(all_similarities) > 0:
            mean_similarity = torch.tensor(
                sum(all_similarities) / len(all_similarities),
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
        else:
            mean_similarity = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # MinMax normalization for visual attention weights
        valid_mask = torch.ones_like(visual_attention_weights, dtype=torch.bool)
        for batch_idx in range(batch_size):
            for gen_idx in range(response_length):
                gen_pos = prompt_length + gen_idx
                if gen_pos >= hidden_states.size(1):
                    valid_mask[batch_idx, gen_idx] = False
        
        if valid_mask.any():
            valid_weights = torch.masked_select(visual_attention_weights, valid_mask)
            
            if valid_weights.numel() > 0:
                global_min = valid_weights.min()
                global_max = valid_weights.max()
                global_range = global_max - global_min + 1e-8
                visual_attention_weights = (visual_attention_weights - global_min) / global_range
                visual_attention_weights[~valid_mask] = 0.0
            else:
                min_val = visual_attention_weights.min(dim=1, keepdim=True)[0]
                max_val = visual_attention_weights.max(dim=1, keepdim=True)[0]
                visual_attention_weights = (visual_attention_weights - min_val) / (max_val - min_val + 1e-8)
        else:
            min_val = visual_attention_weights.min(dim=1, keepdim=True)[0]
            max_val = visual_attention_weights.max(dim=1, keepdim=True)[0]
            visual_attention_weights = (visual_attention_weights - min_val) / (max_val - min_val + 1e-8)
        
        return visual_attention_weights, mean_similarity

    
    def _compute_visual_compensation_factor(
        self,
        response_mask: torch.Tensor,
        strength: float = 0.5,
        start_ratio: float = 0.0,
        modulation_type: str = "linear",
        exponent: float = 2.0,
        step_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute visual attention compensation factors (Eq.5).

        Supports three schedule types:
        1. Linear: C(t) = β * (t/T)  (default, Eq.5)
        2. Exponential: C(t) = β * (t/T)^exponent
        3. Step-function: C(t) = β if t/T >= step_ratio, else 0

        Args:
            response_mask: (batch_size, response_length) - mask for valid response tokens
            strength: compensation intensity β
            start_ratio: start position ratio (0.0 = from beginning)
            modulation_type: schedule type ('linear', 'exponential', or 'step-function')
            exponent: exponent for exponential schedule
            step_ratio: step position ratio for step-function

        Returns:
            compensation: (batch_size, response_length) - compensation factors to add to w_t
        """
        batch_size, response_length = response_mask.shape
        compensation = torch.zeros_like(response_mask, dtype=torch.float32)

        # Create position indices: [0, 1, 2, ..., response_length-1]
        positions = torch.arange(response_length, device=response_mask.device, dtype=torch.float32)
        positions = positions.unsqueeze(0).expand(batch_size, -1)  # (batch_size, response_length)

        for batch_idx in range(batch_size):
            valid_mask = response_mask[batch_idx].bool()
            if not valid_mask.any():
                continue

            valid_positions = positions[batch_idx, valid_mask]
            max_pos = valid_positions.max().item() if valid_positions.numel() > 0 else 1.0
            min_pos = valid_positions.min().item() if valid_positions.numel() > 0 else 0.0
            pos_range = max_pos - min_pos + 1e-8

            normalized_pos = (valid_positions - min_pos) / pos_range
            start_mask = normalized_pos >= start_ratio

            if start_mask.any():
                if modulation_type == "linear":
                    mod_pos = (normalized_pos[start_mask] - start_ratio) / (1.0 - start_ratio + 1e-8)
                    comp_values = strength * mod_pos
                    
                elif modulation_type == "exponential":
                    mod_pos = (normalized_pos[start_mask] - start_ratio) / (1.0 - start_ratio + 1e-8)
                    mod_pos = torch.clamp(mod_pos, min=0.0, max=1.0)
                    comp_values = strength * (mod_pos ** exponent)
                    
                elif modulation_type == "step-function":
                    step_mask = normalized_pos[start_mask] >= step_ratio
                    comp_values = torch.where(
                        step_mask,
                        torch.full_like(normalized_pos[start_mask], strength),
                        torch.zeros_like(normalized_pos[start_mask])
                    )
                else:
                    mod_pos = (normalized_pos[start_mask] - start_ratio) / (1.0 - start_ratio + 1e-8)
                    comp_values = strength * mod_pos

                compensation[batch_idx, valid_mask] = torch.where(
                    start_mask,
                    comp_values,
                    torch.zeros_like(comp_values)
                )

        return compensation


    def _forward_micro_batch(self, micro_batch: dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            output_kwargs = {
                "input_ids": input_ids_rmpad,
                "attention_mask": None,
                "position_ids": position_ids_rmpad,
                **multi_modal_inputs,
                "use_cache": False,
            }
            # Enable output_hidden_states if visual attention is needed
            if getattr(self.config, 'use_intra_trajectory_reweighting', False) or \
               getattr(self.config, 'use_inter_trajectory_reweighting', False):
                output_kwargs["output_hidden_states"] = True
            output = self.actor_module(**output_kwargs)  # prevent model thinks we are generating

            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            entropy = torch.zeros_like(log_probs)
            
            # Compute visual attention weights for padding_free path
            # Needed for intra-trajectory or inter-trajectory reweighting
            visual_attention_weights = None
            mean_similarity = None
            need_visual_attention = (
                getattr(self.config, 'use_intra_trajectory_reweighting', False) or
                getattr(self.config, 'use_inter_trajectory_reweighting', False)
            )
            if need_visual_attention:
                # Try to get hidden_states from output
                hidden_states = None
                if hasattr(output, 'hidden_states') and output.hidden_states is not None:
                    hidden_states = output.hidden_states
                elif hasattr(output, 'last_hidden_state'):
                    # Some models return last_hidden_state instead
                    hidden_states = [output.last_hidden_state]
                elif isinstance(output, tuple) and len(output) > 0:
                    # If output is a tuple, the first element might be hidden_states
                    # But we need the last layer, so we'll need to get it differently
                    # For now, try to access it from the model's base output
                    pass
                
                if hidden_states is None:
                    if self.rank == 0:
                        print("WARNING: output.hidden_states is None. Visual attention weights cannot be computed.")
                        print(f"       Trying alternative: hasattr(output, 'last_hidden_state'): {hasattr(output, 'last_hidden_state')}")
                        if hasattr(output, 'last_hidden_state'):
                            print(f"       output.last_hidden_state shape: {output.last_hidden_state.shape if output.last_hidden_state is not None else None}")
                else:
                    # Get hidden states and pad back
                    # hidden_states is a tuple/list of all layers, we need the last one
                    if isinstance(hidden_states, (tuple, list)):
                        last_hidden_state = hidden_states[-1]
                    else:
                        last_hidden_state = hidden_states
                    hidden_states_rmpad = last_hidden_state.squeeze(0)  # (total_nnz, hidden_dim)
                    
                    if self.config.ulysses_size > 1:
                        hidden_states_rmpad = gather_outputs_and_unpad(
                            hidden_states_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    
                    # Pad back to (bsz, seqlen, hidden_dim)
                    full_hidden_states = pad_input(
                        hidden_states=hidden_states_rmpad.unsqueeze(-1), 
                        indices=indices, 
                        batch=batch_size, 
                        seqlen=seqlen
                    ).squeeze(-1)  # (bsz, seqlen, hidden_dim)
                    
                    prompt_length = seqlen - response_length
                    visual_attention_result = self._compute_visual_attention_weights(
                        full_hidden_states, input_ids, response_length, prompt_length
                    )
                    if visual_attention_result is not None:
                        visual_attention_weights, mean_similarity = visual_attention_result
                        if self.rank == 0 and not getattr(self, '_visual_attn_first_call', False):
                            print(f"[Visual Attention] Computed weights shape: {visual_attention_weights.shape}, "
                                  f"mean: {visual_attention_weights.mean().item():.4f}, "
                                  f"min: {visual_attention_weights.min().item():.4f}, "
                                  f"max: {visual_attention_weights.max().item():.4f}, "
                                  f"mean_similarity: {mean_similarity.item():.4f}")
                            self._visual_attn_first_call = True
                    else:
                        visual_attention_weights = None
                        mean_similarity = None
        else:
            output_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                **multi_modal_inputs,
                "use_cache": False,
            }
            # Enable output_hidden_states if visual attention is needed
            if getattr(self.config, 'use_intra_trajectory_reweighting', False) or \
               getattr(self.config, 'use_inter_trajectory_reweighting', False):
                output_kwargs["output_hidden_states"] = True
            
            output = self.actor_module(**output_kwargs)

            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

            entropy = torch.zeros_like(log_probs)

            # Compute visual attention weights for non-padding_free path
            # Needed for intra-trajectory or inter-trajectory reweighting
            visual_attention_weights = None
            mean_similarity = None
            need_visual_attention = (
                getattr(self.config, 'use_intra_trajectory_reweighting', False) or
                getattr(self.config, 'use_inter_trajectory_reweighting', False)
            )
            if need_visual_attention:
                # Try to get hidden_states from output
                hidden_states = None
                if hasattr(output, 'hidden_states') and output.hidden_states is not None:
                    hidden_states = output.hidden_states
                elif hasattr(output, 'last_hidden_state') and output.last_hidden_state is not None:
                    # Some models return last_hidden_state instead of hidden_states tuple
                    hidden_states = [output.last_hidden_state]
                
                if hidden_states is None:
                    if self.rank == 0:
                        print("WARNING: output.hidden_states is None. Visual attention weights cannot be computed.")
                        print(f"       Trying alternative: hasattr(output, 'last_hidden_state'): {hasattr(output, 'last_hidden_state')}")
                else:
                    # Get the last layer hidden states
                    if isinstance(hidden_states, (tuple, list)):
                        last_hidden_state = hidden_states[-1]  # (batch, seq_len, hidden_dim)
                    else:
                        last_hidden_state = hidden_states
                    prompt_length = input_ids.size(1) - response_length
                    visual_attention_result = self._compute_visual_attention_weights(
                        last_hidden_state, input_ids, response_length, prompt_length
                    )
                    if visual_attention_result is not None:
                        visual_attention_weights, mean_similarity = visual_attention_result
                        if self.rank == 0 and not getattr(self, '_visual_attn_first_call', False):
                            print(f"[Visual Attention] Computed weights shape: {visual_attention_weights.shape}, "
                                  f"mean: {visual_attention_weights.mean().item():.4f}, "
                                  f"min: {visual_attention_weights.min().item():.4f}, "
                                  f"max: {visual_attention_weights.max().item():.4f}, "
                                  f"mean_similarity: {mean_similarity.item():.4f}")
                            self._visual_attn_first_call = True
                    else:
                        visual_attention_weights = None
                        mean_similarity = None
        return {
            "log_probs": log_probs, 
            "entropy": entropy,
            "visual_attention_weights": visual_attention_weights,
            "visual_attention_mean_similarity": mean_similarity,
        }


    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)['log_probs']
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        non_tensor_select_keys = ["multi_modal_inputs"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # Outputs from the forward pass are shaped (bsz, response_length).
                    output = self._forward_micro_batch(model_inputs, temperature=temperature)
                    log_probs = output['log_probs']
                    entropy = output['entropy']
                    visual_attention_weights = output.get('visual_attention_weights', None)
                    visual_attention_mean_similarity = output.get('visual_attention_mean_similarity', None)

                    loss_token_mask = None # Default to None
                    
                    loss_token_weights = None # Default to None (for soft masking)
                    advantage_weighting_factors = None # Default to None (for advantage weighting in PG framework)
                    intra_weights = None # Initialize for trajectory-level use

                    # ============================================================
                    # Token-Level Visual Weighting
                    # ============================================================
                    
                    # Intra-trajectory reweighting: compute normalized visual attention weights
                    if self.config.use_intra_trajectory_reweighting:
                        if visual_attention_weights is None:
                            if self.rank == 0 and not getattr(self, '_intra_reweighting_warning_shown', False):
                                print("WARNING: use_intra_trajectory_reweighting is enabled but visual_attention_weights is None. "
                                      "This may happen if there are no visual tokens in the input.")
                                self._intra_reweighting_warning_shown = True
                        else:
                            # Start with raw visual attention weights (w_t, not normalized)
                            intra_weights = visual_attention_weights.clone()
                            
                            # Mask invalid tokens (non-response tokens)
                            intra_weights[~response_mask.bool()] = 0.0
                            
                            # Visual attention compensation (Eq.5-6)
                            # w_t * [1 + G_i(ρ) * β * (t/T)]
                            visual_compensation = None
                            gating_factors = None
                            if getattr(self.config, 'use_visual_compensation', False):
                                vc_strength = getattr(self.config, 'visual_compensation_strength', 0.3)
                                vc_start_ratio = getattr(self.config, 'visual_compensation_start_ratio', 0.0)
                                vc_type = getattr(self.config, 'visual_compensation_type', 'linear')
                                vc_exponent = getattr(self.config, 'visual_compensation_exponent', 2.0)
                                vc_step_ratio = getattr(self.config, 'visual_compensation_step_ratio', 0.5)

                                visual_compensation = self._compute_visual_compensation_factor(
                                    response_mask,
                                    strength=vc_strength,
                                    start_ratio=vc_start_ratio,
                                    modulation_type=vc_type,
                                    exponent=vc_exponent,
                                    step_ratio=vc_step_ratio,
                                )

                                if self.rank == 0 and not getattr(self, '_vac_executed', False):
                                    vc_valid = visual_compensation[response_mask.bool()]
                                    print(f"[VGPO] Visual Attention Compensation (Eq.5) EXECUTED: "
                                          f"β={vc_strength}, schedule={vc_type}, "
                                          f"compensation_mean={vc_valid.mean().item():.4f}, compensation_max={vc_valid.max().item():.4f}")
                                    self._vac_executed = True

                                # Gated visual compensation (Eq.6)
                                use_gating = getattr(self.config, 'use_gated_visual_compensation', False)
                                if use_gating:
                                    visual_threshold = getattr(self.config, 'visual_attention_threshold', 0.4)
                                    gated_start_ratio = getattr(self.config, 'gated_visual_compensation_start_ratio', 0.5)
                                    
                                    # Initialize gating factors to zero (use float32 to avoid dtype mismatch)
                                    gating_factors = torch.zeros_like(intra_weights, dtype=torch.float32)
                                    
                                    # For each sequence, find Top-k% tokens based on absolute w_t values
                                    # Only consider tokens from gated_start_ratio onwards
                                    for batch_idx in range(intra_weights.size(0)):
                                        valid_mask = response_mask[batch_idx].bool()
                                        if valid_mask.any():
                                            valid_weights = intra_weights[batch_idx, valid_mask]
                                            if valid_weights.numel() > 0:
                                                # Determine the start position for Top-k% selection
                                                num_valid = valid_weights.numel()
                                                start_idx = int(num_valid * gated_start_ratio)
                                                
                                                # Only consider tokens from start_idx onwards
                                                if start_idx < num_valid:
                                                    # Get weights in the range [start_idx, num_valid)
                                                    range_weights = valid_weights[start_idx:]
                                                    num_range = len(range_weights)
                                                    
                                                    if num_range > 0:
                                                        # Calculate the number of tokens in Top-k% within this range
                                                        num_top_k = max(1, int(num_range * visual_threshold))  # At least 1 token
                                                        
                                                        # Find the threshold value for Top-k% within this range
                                                        # Sort in descending order and take the k-th percentile value
                                                        sorted_weights, _ = torch.sort(range_weights, descending=True)
                                                        threshold_value = sorted_weights[num_top_k - 1]  # k-th largest value in range
                                                        
                                                        # Set gating_factors = 1 for Top-k% tokens in the range, 0 for others
                                                        # For tokens before start_idx, gating_factors remains 0
                                                        range_top_k_mask = range_weights >= threshold_value
                                                        
                                                        # Fix chained indexing issue in PyTorch
                                                        valid_indices = torch.nonzero(valid_mask).squeeze(-1)
                                                        range_indices = valid_indices[start_idx:]
                                                        gating_factors[batch_idx, range_indices] = range_top_k_mask.to(gating_factors.dtype)
                                    
                                    gated_compensation = gating_factors * visual_compensation
                                    intra_weights = intra_weights * (1.0 + gated_compensation)

                                    if self.rank == 0 and not getattr(self, '_gated_vac_executed', False):
                                        gate_rate = gating_factors[response_mask.bool()].mean().item()
                                        print(f"[VGPO] Gated Visual Compensation (Eq.6) EXECUTED: "
                                              f"γ={gated_start_ratio}, κ={visual_threshold}, gate_pass_rate={gate_rate:.4f}")
                                        self._gated_vac_executed = True
                                else:
                                    intra_weights = intra_weights * (1.0 + visual_compensation)
                            
                            # MinMax normalization per sequence (Eq. 7)
                            for batch_idx in range(intra_weights.size(0)):
                                valid_mask = response_mask[batch_idx].bool()
                                if valid_mask.any():
                                    valid_weights = intra_weights[batch_idx, valid_mask]
                                    if valid_weights.numel() > 0:
                                        w_min = valid_weights.min()
                                        w_max = valid_weights.max()
                                        w_range = w_max - w_min + 1e-8
                                        intra_weights[batch_idx, valid_mask] = (valid_weights - w_min) / w_range
                            intra_weights[~response_mask.bool()] = 0.0
                            
                            # Advantage weighting (Eq. 8): ψ_{i,t} = w_hat_{i,t} - mean(w_hat)
                            if True:
                                # Credit assignment in Policy Gradient framework
                                # Use mean-centered adjustment: ψ(w_t) = 1 + (w_t - w_avg)
                                # This preserves the average advantage scale while amplifying/suppressing based on visual attention
                                
                                # Compute per-sequence mean weight for valid tokens
                                advantage_weighting_factors = torch.ones_like(intra_weights)
                                
                                for batch_idx in range(intra_weights.size(0)):
                                    valid_mask = response_mask[batch_idx].bool()
                                    if valid_mask.any():
                                        valid_weights = intra_weights[batch_idx, valid_mask]
                                        if valid_weights.numel() > 0:
                                            w_avg = valid_weights.mean()
                                            # Apply mean-centered adjustment: ψ(w_t) = 1 + (w_t - w_avg)
                                            advantage_weighting_factors[batch_idx, valid_mask] = 1.0 + (valid_weights - w_avg)
                                
                                # Clamp to reasonable range to avoid extreme values
                                advantage_weighting_factors = torch.clamp(advantage_weighting_factors, min=0.1, max=2.0)
                                advantage_weighting_factors[~response_mask.bool()] = 1.0  # No change for invalid tokens
                                
                            if self.rank == 0 and not getattr(self, '_intra_reweighting_executed', False):
                                weight_mean = intra_weights[response_mask.bool()].mean().item()
                                factor_mean = advantage_weighting_factors[response_mask.bool()].mean().item()
                                factor_min = advantage_weighting_factors[response_mask.bool()].min().item()
                                factor_max = advantage_weighting_factors[response_mask.bool()].max().item()
                                print(f"[VGPO] Intra-Trajectory Reweighting (Eq.7-8) EXECUTED: "
                                      f"weight_mean={weight_mean:.4f}, ψ_mean={factor_mean:.4f}, ψ_range=[{factor_min:.4f}, {factor_max:.4f}]")
                                self._intra_reweighting_executed = True
                            
                            # Log metrics
                            with torch.no_grad():
                                valid_weights = intra_weights[response_mask.bool()]
                                if valid_weights.numel() > 0:
                                    metrics["actor/intra_weight_mean"].append(valid_weights.mean().item())
                                    metrics["actor/intra_weight_min"].append(valid_weights.min().item())
                                    metrics["actor/intra_weight_max"].append(valid_weights.max().item())
                                    metrics["actor/intra_weight_std"].append(valid_weights.std().item())

                                # Compute visual token distribution across two segments (Top-40% based on visual similarity)
                                # This metric tracks how visual tokens are distributed between first half and second half
                                # The two halves' visual token percentages should sum to 40% (visual_threshold_for_dist)
                                visual_threshold_for_dist = 0.4  # Top-40%
                                early_visual_token_percentages = []  # Percentage of total sequence
                                late_visual_token_percentages = []  # Percentage of total sequence
                                
                                for batch_idx in range(response_mask.size(0)):
                                    valid_mask = response_mask[batch_idx].bool()
                                    if valid_mask.sum() > 0:
                                        # Get raw visual attention weights (before any position modulation)
                                        if visual_attention_weights is not None:
                                            valid_visual_weights = visual_attention_weights[batch_idx, valid_mask]
                                            
                                            if valid_visual_weights.numel() > 0:
                                                num_valid = valid_visual_weights.numel()
                                                
                                                # Find Top-40% tokens based on visual similarity
                                                num_top_k = max(1, int(num_valid * visual_threshold_for_dist))
                                                sorted_weights, _ = torch.sort(valid_visual_weights, descending=True)
                                                threshold_value = sorted_weights[num_top_k - 1]
                                                
                                                # Identify visual tokens (Top-40%)
                                                visual_token_mask = valid_visual_weights >= threshold_value
                                                
                                                # Split sequence into two halves
                                                half_point = num_valid // 2
                                                
                                                # First half visual tokens (as percentage of total sequence)
                                                first_half_mask = visual_token_mask[:half_point]
                                                first_half_visual_count = first_half_mask.sum().item()
                                                first_half_percentage = first_half_visual_count / (num_valid + 1e-8)  # Percentage of total
                                                
                                                # Second half visual tokens (as percentage of total sequence)
                                                second_half_mask = visual_token_mask[half_point:]
                                                second_half_visual_count = second_half_mask.sum().item()
                                                second_half_percentage = second_half_visual_count / (num_valid + 1e-8)  # Percentage of total
                                                
                                                # Store percentages (should sum to approximately 40%)
                                                early_visual_token_percentages.append(first_half_percentage)
                                                late_visual_token_percentages.append(second_half_percentage)
                                
                                # Log batch-level metrics
                                if len(early_visual_token_percentages) > 0:
                                    metrics["actor/visual_token_percentage_first_half"].append(
                                        sum(early_visual_token_percentages) / len(early_visual_token_percentages)
                                    )
                                if len(late_visual_token_percentages) > 0:
                                    metrics["actor/visual_token_percentage_second_half"].append(
                                        sum(late_visual_token_percentages) / len(late_visual_token_percentages)
                                    )
                                
                                # Log the sum (should be close to 40%)
                                if len(early_visual_token_percentages) > 0 and len(late_visual_token_percentages) > 0:
                                    avg_early = sum(early_visual_token_percentages) / len(early_visual_token_percentages)
                                    avg_late = sum(late_visual_token_percentages) / len(late_visual_token_percentages)
                                    metrics["actor/visual_token_percentage_total"].append(avg_early + avg_late)
                                    metrics["actor/visual_token_percentage_late_early_ratio"].append(
                                        avg_late / (avg_early + 1e-8)
                                    )
                                
                                # Log visual compensation metrics
                                if visual_compensation is not None:
                                    valid_vc = visual_compensation[response_mask.bool()]
                                    if valid_vc.numel() > 0:
                                        metrics["actor/visual_compensation_mean"].append(valid_vc.mean().item())
                                        metrics["actor/visual_compensation_max"].append(valid_vc.max().item())
                                        metrics["actor/visual_compensation_std"].append(valid_vc.std().item())

                                        early_vc_list = []
                                        late_vc_list = []
                                        for batch_idx in range(response_mask.size(0)):
                                            valid_mask = response_mask[batch_idx].bool()
                                            if valid_mask.sum() > 0:
                                                valid_vc_batch = visual_compensation[batch_idx, valid_mask]

                                                if len(valid_vc_batch) >= 3:
                                                    third = len(valid_vc_batch) // 3
                                                    early_vc_list.append(valid_vc_batch[:third])
                                                    late_vc_list.append(valid_vc_batch[-third:])

                                        if len(early_vc_list) > 0:
                                            all_early = torch.cat(early_vc_list)
                                            metrics["actor/visual_compensation_early_mean"].append(all_early.mean().item())
                                        if len(late_vc_list) > 0:
                                            all_late = torch.cat(late_vc_list)
                                            metrics["actor/visual_compensation_late_mean"].append(all_late.mean().item())
                                            if len(early_vc_list) > 0:
                                                metrics["actor/visual_compensation_late_early_ratio"].append(
                                                    all_late.mean().item() / (all_early.mean().item() + 1e-8)
                                                )

                                # Log gating metrics if enabled
                                if gating_factors is not None:
                                    valid_gating = gating_factors[response_mask.bool()]
                                    if valid_gating.numel() > 0:
                                        metrics["actor/gating_factor_mean"].append(valid_gating.mean().item())
                                        metrics["actor/gating_factor_min"].append(valid_gating.min().item())
                                        metrics["actor/gating_factor_max"].append(valid_gating.max().item())
                                        metrics["actor/gating_factor_std"].append(valid_gating.std().item())
                                        
                                        # Log fraction of tokens that pass the gate (gating > 0.5)
                                        gating_passed = (valid_gating > 0.5).float()
                                        metrics["actor/gating_pass_rate"].append(gating_passed.mean().item())

                                        # Log position-specific gating pass rates: early vs late tokens
                                        # Collect early and late gating factors across all sequences in the batch
                                        early_gating_list = []
                                        late_gating_list = []
                                        early_gating_passed_list = []
                                        late_gating_passed_list = []
                                        
                                        for batch_idx in range(response_mask.size(0)):
                                            valid_mask = response_mask[batch_idx].bool()
                                            if valid_mask.sum() > 0:
                                                valid_gating_batch = gating_factors[batch_idx, valid_mask]
                                                
                                                if len(valid_gating_batch) >= 3:
                                                    third = len(valid_gating_batch) // 3
                                                    early_gating_list.append(valid_gating_batch[:third])
                                                    late_gating_list.append(valid_gating_batch[-third:])
                                                    
                                                    # Compute pass rate for early and late tokens
                                                    early_passed = (valid_gating_batch[:third] > 0.5).float()
                                                    late_passed = (valid_gating_batch[-third:] > 0.5).float()
                                                    early_gating_passed_list.append(early_passed)
                                                    late_gating_passed_list.append(late_passed)
                                        
                                        # Compute batch-level early/late pass rates
                                        if len(early_gating_passed_list) > 0:
                                            all_early_passed = torch.cat(early_gating_passed_list)
                                            metrics["actor/gating_pass_rate_early"].append(all_early_passed.mean().item())
                                        if len(late_gating_passed_list) > 0:
                                            all_late_passed = torch.cat(late_gating_passed_list)
                                            metrics["actor/gating_pass_rate_late"].append(all_late_passed.mean().item())
                                            
                                            # Also log the ratio: late / early pass rate
                                            if len(early_gating_passed_list) > 0:
                                                metrics["actor/gating_pass_rate_late_early_ratio"].append(
                                                    all_late_passed.mean().item() / (all_early_passed.mean().item() + 1e-8)
                                                )
                                
                                if advantage_weighting_factors is not None:
                                    valid_factors = advantage_weighting_factors[response_mask.bool()]
                                    if valid_factors.numel() > 0:
                                        metrics["actor/advantage_weighting_factor_mean"].append(valid_factors.mean().item())
                                        metrics["actor/advantage_weighting_factor_min"].append(valid_factors.min().item())
                                        metrics["actor/advantage_weighting_factor_max"].append(valid_factors.max().item())
                                        metrics["actor/advantage_weighting_factor_std"].append(valid_factors.std().item())
                    
                    # ============================================================
                    # Trajectory-Level Visual Weighting
                    # ============================================================
                    # This section handles trajectory-level visual attention weighting.
                    # It uses token-level computed intra_weights to calculate trajectory scores.
                    # Within each rollout group, it applies minmax normalization and
                    # mean-centered adjustment: α(S(τ_i)) = 1 + (S_norm - S_avg)
                    # ============================================================
                    
                    # Apply trajectory-level visual weighting if enabled (within rollout groups)
                    inter_reweighting_factors = None
                    if getattr(self.config, 'use_inter_trajectory_reweighting', False):
                        traj_inter_weights = None
                        if self.config.use_intra_trajectory_reweighting and intra_weights is not None:
                            traj_inter_weights = intra_weights
                        elif visual_attention_weights is not None:
                            traj_inter_weights = visual_attention_weights.clone()
                            traj_inter_weights[~response_mask.bool()] = 0.0
                            # Apply simple normalization to [0, 1] per sequence
                            for batch_idx in range(traj_inter_weights.size(0)):
                                valid_mask = response_mask[batch_idx].bool()
                                if valid_mask.any():
                                    valid_weights = traj_inter_weights[batch_idx, valid_mask]
                                    if valid_weights.numel() > 0:
                                        w_min = valid_weights.min()
                                        w_max = valid_weights.max()
                                        w_range = w_max - w_min + 1e-8
                                        traj_inter_weights[batch_idx, valid_mask] = (valid_weights - w_min) / w_range
                            traj_inter_weights[~response_mask.bool()] = 0.0
                        
                        if traj_inter_weights is not None:
                            # Create index based on batch size (sequential indices within micro-batch)
                            batch_size = advantages.shape[0]
                            index = torch.arange(batch_size, device=advantages.device, dtype=torch.long)

                            rollout_n = getattr(self.config, 'rollout_n', 8)

                            inter_reweighting_factors = compute_inter_trajectory_reweighting(
                                traj_inter_weights,
                                response_mask,
                                index,
                                rollout_n=rollout_n,
                                normalization='minmax',
                                temperature=1.0,
                                clamp_min=0.9,
                                clamp_max=2.0,
                            )

                            # Apply trajectory weighting to advantages (Eq. 11)
                            advantages = advantages * inter_reweighting_factors.unsqueeze(1)

                            if self.rank == 0 and not getattr(self, '_inter_reweighting_executed', False):
                                print(f"[VGPO] Inter-Trajectory Reweighting (Eq.9-11) EXECUTED: "
                                      f"rollout_n={rollout_n}, "
                                      f"α_mean={inter_reweighting_factors.mean().item():.4f}, "
                                      f"α_range=[{inter_reweighting_factors.min().item():.4f}, {inter_reweighting_factors.max().item():.4f}]")
                                self._inter_reweighting_executed = True

                            # Log metrics
                            with torch.no_grad():
                                metrics["actor/trajectory_weighting_factor_mean"].append(
                                    inter_reweighting_factors.mean().item()
                                )
                                metrics["actor/trajectory_weighting_factor_min"].append(
                                    inter_reweighting_factors.min().item()
                                )
                                metrics["actor/trajectory_weighting_factor_max"].append(
                                    inter_reweighting_factors.max().item()
                                )
                                metrics["actor/advantage_after_trajectory_weighting_mean"].append(
                                    advantages[response_mask.bool()].mean().item()
                                )
                        else:
                            if self.rank == 0 and not getattr(self, '_trajectory_weighting_warning_shown', False):
                                print("WARNING: use_inter_trajectory_reweighting is enabled but intra_weights and visual_attention_weights are both None.")
                                self._trajectory_weighting_warning_shown = True


                    # Apply advantage weighting if enabled (credit assignment in PG framework)
                    if advantage_weighting_factors is not None:
                        # Weight advantages: advantages_weighted = advantages * ψ(w_t)
                        # This directly affects how much each action is reinforced or suppressed
                        advantages = advantages * advantage_weighting_factors

                        # if self.rank == 0: print(advantage_weighting_factors)
                            
                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                        loss_token_mask=loss_token_mask,
                        loss_token_weights=loss_token_weights
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef
                    else:
                        loss = pg_loss

                    if self.config.use_entropy_penalty:
                        # Use entropy penalty for training
                        entropy_loss = -VF.masked_mean(log_probs, response_mask)
                        loss = loss + entropy_loss * self.config.entropy_penalty_coef
                        metrics["actor/entropy_penalty_coef"] = self.config.entropy_penalty_coef
                    
                    # Use selected tokens for scaling when loss_token_mask is used. This prevents reward hacking where model learns to shorten responses
                    if loss_token_mask is not None:
                        total_selected_tokens = torch.sum(loss_token_mask)
                        dist.all_reduce(total_selected_tokens, op=dist.ReduceOp.SUM)
                        loss = loss * torch.sum(loss_token_mask) * self.world_size / total_selected_tokens
                    else:
                        loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    # loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    # loss.backward()

                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
