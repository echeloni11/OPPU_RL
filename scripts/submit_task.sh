#!/usr/bin/env bash
set -euo pipefail

#########################
# ==== 配置区 =====
PARTITION=general
GPU_TYPE=a100
GPU_COUNT=1
TIME_LIMIT=12:00:00          # 12 h
MEM=50G
CPUS=4

CONDA_ENV=oppu
PROJECT_DIR="$HOME/projects/OPPU_RL"
LOG_DIR="$PROJECT_DIR/logs"
TASKS=(citation movie_tagging news_categorize news_headline product_rating scholarly_title tweet_paraphrase)
#########################

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# ------- 提交函数 -------
submit_job () {
  local job_prefix="$1"       # OPPU 或 GRPO
  local task="$2"
  local script="$3"           # .py 文件

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
python task_LoRA.py --k 0 --task_name "$task"
EOF
}

# ------- 主循环 -------
for task in "${TASKS[@]}"; do
  submit_job "Base" "${task}"       "task_LoRA.py"
done
