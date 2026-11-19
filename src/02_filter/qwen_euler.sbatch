#!/bin/bash
#SBATCH --job-name=ideology-qwen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --gpus=rtx_6000:1
#SBATCH --output=ideology-qwen-%j.out

set -euo pipefail
module load stack/2024-06 python_cuda/3.11.6 eth_proxy
source ~/venvs/political-pairs/bin/activate
python ~/01_qwen_filter.py
