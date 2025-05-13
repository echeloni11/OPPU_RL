#!/usr/bin/env bash
#
#SBATCH --job-name=LoRA_batch
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --output=logs/LoRA_%j.out
#SBATCH --error=logs/LoRA_%j.err

# 1. 切到项目目录（已经由 --chdir 自动完成）
echo "工作目录：$(pwd)"

# 2. 加载 CUDA，并激活 conda 环境
conda activate oppu

echo "开始时间：$(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "使用 CPU 核心：$SLURM_CPUS_PER_TASK"
echo "分配内存：$SLURM_MEM_PER_NODE"

# 3. 定义任务列表
tasks=(
  citation
  movie_tagging
  news_categorize
  news_headline
  product_rating
  scholarly_title
  tweet_paraphrase
)

for task_name in "${tasks[@]}"; do
  echo "=== 运行任务: $task_name ==="
  # 利用 srun 启动一个新的 job step，python 进程结束后会自动释放 GPU
  srun python task_LoRA.py --k 0 --task_name "$task_name"
  echo "=== 完成任务: $task_name ==="

done

echo "所有任务完成，结束时间：$(date)"
