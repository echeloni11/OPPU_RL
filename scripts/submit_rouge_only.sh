#!/usr/bin/env bash
# submit_all.sh  ——  批量提交 OPPU / GRPO 训练任务
set -euo pipefail

#########################
# ==== 配置区 =====
PARTITION=general
GPU_TYPE=a100
GPU_COUNT=1
TIME_LIMIT=12:00:00          # 12 h
MEM=100G
CPUS=4

CONDA_ENV=oppu
PROJECT_DIR="$HOME/projects/OPPU_RL"
LOG_DIR="$PROJECT_DIR/logs"
TASKS=(news_headline scholarly_title)
#########################

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# ------- 提交函数 -------
submit_job () {
  local job_prefix="$1"       # OPPU 或 GRPO
  local task="$2"
  local script="$3"           # .py 文件
  local num_users_arg=""

  if [[ "$task" == "news_headline" || "$task" == "product_rating" ]]; then
    num_users_arg="--num_users 10"
  elif [[ "$task" == "scholarly_title" ]]; then
    num_users_arg="--num_users 10"
  elif [[ "$task" == "news_categorize" ]]; then
    num_users_arg="--num_users 10"
  fi

  sbatch <<-EOF
#!/bin/bash
#SBATCH --job-name=${job_prefix}_${task}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPU_TYPE}:${GPU_COUNT}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH -o ${LOG_DIR}/${job_prefix}_${task}-%j.out
#SBATCH -e ${LOG_DIR}/${job_prefix}_${task}-%j.err

# 载入 conda
conda activate ${CONDA_ENV}

cd ${PROJECT_DIR}

python ${script} \
  --k 0 --task_name ${task} \
  --task_lora ./ckpt/${task}/k0-${task}-Llama-2-7b-hf-task_LoRA_ckpt \
  ${num_users_arg}
EOF
}

# ------- 主循环 -------
for task in "${TASKS[@]}"; do
  submit_job "GRPO" "${task}"       "OPPU_GRPO.py"
done
