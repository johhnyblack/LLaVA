#PBS -N llava_vizwiz_eval
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=01:00:00
#PBS -p -500
#PBS -o /logs/output.log

# 设置工作目录
cd /home/users/nus/ophv119/LLaVA

# 创建日志目录
mkdir -p logs

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/users/nus/ophv119/LLaVA:$PYTHONPATH
export TRANSFORMERS_CACHE=/scratch/users/nus/ophv119/llava_model/
export HF_HOME=/scratch/users/nus/ophv119/llava_model/
export HF_HUB_CACHE=/scratch/users/nus/ophv119/llava_model/

mkdir -p /scratch/users/nus/ophv119/llava_model/

source /home/users/nus/ophv119/scratch/miniconda3/etc/profile.d/conda.sh
conda activate llava
echo $(python -c "import torch; print(torch.__version__)") > logs/vizwiz_model_eval.log | tee logs/vizwiz_model_eval.log

# 创建输出目录
mkdir -p ./playground/data/eval/vizwiz/answers
mkdir -p ./playground/data/eval/vizwiz/answers_upload

# 执行第一个命令并实时输出
echo "Starting LLaVA VizWiz evaluation at $(date)"
echo "================================================"

python -m llava.eval.model_vqa_loader \
    --model-path /home/users/nus/ophv119/LLaVA/checkpoints/llava-v1.5-13b-sdc-lora \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 2>&1 | tee logs/vizwiz_model_eval.log

echo "Model evaluation completed at $(date)"
echo "================================================"

# 执行第二个命令并实时输出
echo "Starting format conversion at $(date)"

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-v1.5-13b.json 2>&1 | tee logs/vizwiz_convert.log

echo "Format conversion completed at $(date)"
echo "================================================"
echo "All tasks completed successfully at $(date)"