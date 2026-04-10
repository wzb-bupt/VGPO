#!/bin/bash

# set -x

export SWANLAB_API_KEY="your swanlab key"
export SWANLAB_LOG_DIR="logs/swanlab/"

EXP_NAME="VGPO_7B_VIRL39K"

# ----------------------------------
# Model & DataSet
# ----------------------------------
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
TRAIN_FILE=PAPOGalaxy/PAPO_ViRL39K_train
VAL_FILE=PAPOGalaxy/PAPO_MMK12_test
ROLLOUT_N=8
EPOCH=2

# ----------------------------------
# Global Parameters
# ----------------------------------
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=512
VAL_BATCH_SIZE=1024
MAX_PROMPT_LENGTH=4096

# ----------------------------------
# VGPO Parameters
# ----------------------------------
use_intra_trajectory_reweighting=true   # Eq.7-8: token-level advantage reweighting by visual focus score ψ_{i,t}
use_inter_trajectory_reweighting=true   # Eq.9-11: trajectory-level advantage reweighting by visual accumulation ϕ_i

use_visual_compensation=true            # Eq.5: visual attention compensation with linear schedule β·(t/T)
use_gated_visual_compensation=true      # Eq.6: visual gate G_i(ρ) to selectively compensate late-stage tokens
gated_visual_compensation_start_ratio=0.5  # γ in Eq.6: tail ratio, only tokens after (1-γ)·T are candidates
visual_compensation_strength=0.3        # β in Eq.5: compensation intensity
visual_attention_threshold=0.2          # κ in Eq.6: top-κ% threshold for visual gate activation

# ----------------------------------
# Main Script
# ----------------------------------
FORMAT_PROMPT="examples/format_prompt/math_format_perception.jinja"
REWARD_FUNCTION="examples/reward_function/math.py:compute_score_wo_format"
CONGI_FILE="examples/configs/config.yaml"

echo "Model Path: ${MODEL_PATH}"
echo "Train File: ${TRAIN_FILE}"
echo "Val File: ${VAL_FILE}"
echo "EXP Name: ${EXP_NAME}"
python3 -m verl.trainer.main \
        config=${CONGI_FILE} \
        data.train_files=${TRAIN_FILE} \
        data.val_files=${VAL_FILE} \
        data.format_prompt=${FORMAT_PROMPT} \
        data.max_prompt_length=${MAX_PROMPT_LENGTH} \
        data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
        worker.actor.model.model_path=${MODEL_PATH} \
        worker.actor.dynamic_batching=True \
        worker.actor.optim.lr=1e-6 \
        worker.rollout.n=${ROLLOUT_N} \
        worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
        worker.reward.reward_function=${REWARD_FUNCTION} \
        worker.actor.micro_batch_size_per_device_for_update=4 \
        worker.actor.micro_batch_size_per_device_for_experience=16 \
        trainer.project_name="VGPO" \
        trainer.experiment_name=${EXP_NAME} \
        trainer.logger=['console','swanlab'] \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        trainer.total_epochs=${EPOCH} \
        algorithm.use_intra_trajectory_reweighting=${use_intra_trajectory_reweighting} \
        algorithm.use_inter_trajectory_reweighting=${use_inter_trajectory_reweighting} \
        algorithm.use_visual_compensation=${use_visual_compensation} \
        algorithm.use_gated_visual_compensation=${use_gated_visual_compensation} \
        algorithm.gated_visual_compensation_start_ratio=${gated_visual_compensation_start_ratio} \
        algorithm.visual_compensation_strength=${visual_compensation_strength} \
        algorithm.visual_attention_threshold=${visual_attention_threshold}
