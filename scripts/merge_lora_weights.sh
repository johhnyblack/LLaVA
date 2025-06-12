#PBS -N merge_lora_weights
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=24:00:00
#PBS -e ./logs/merge_lora_weights.err
#PBS -o ./logs/merge_lora_weights.log

# 设置工作目录
cd /home/users/nus/ophv119/LLaVA


# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/users/nus/ophv119/LLaVA:$PYTHONPATH
export TRANSFORMERS_CACHE=/scratch/users/nus/ophv119/llava_model/
export HF_HOME=/scratch/users/nus/ophv119/llava_model/
export HF_HUB_CACHE=/scratch/users/nus/ophv119/llava_model/

source ~/scratch/miniconda3/etc/profile.d/conda.sh
conda activate llava


python /home/users/nus/ophv119/LLaVA/scripts/merge_lora_weights.py \
    --model-base liuhaotian/llava-v1.5-13b \
    --model-path /home/users/nus/ophv119/LLaVA/checkpoints/llava-v1.5-13b-sdc-lora \
    --save-model-path /home/users/nus/ophv119/LLaVA/checkpoints/llava-v1.5-13b-sdc-lora 2>&1 | tee ./logs/merge_lora_weights.log