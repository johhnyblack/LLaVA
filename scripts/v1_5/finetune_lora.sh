#PBS -N lora_finetune
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /logs/output.log

# 设置工作目录
cd /home/users/nus/ophv119/LLaVA

# 创建日志目录
mkdir -p logs

source ~/bashrc
source /home/users/nus/ophv119/scratch/miniconda3/etc/profile.d/conda.sh
conda activate llava

# 创建 curand 库的符号链接到标准位置
ln -sf /home/users/nus/ophv119/scratch/miniconda3/envs/llava/targets/x86_64-linux/lib/libcurand.so.10.3.6.82 $CONDA_PREFIX/lib/libcurand.so
ln -sf /home/users/nus/ophv119/scratch/miniconda3/envs/llava/targets/x86_64-linux/lib/libcurand.so.10.3.6.82 $CONDA_PREFIX/lib/libcurand.so.10
# 添加到 lib64 目录（如果存在）
mkdir -p $CONDA_PREFIX/lib64
ln -sf /home/users/nus/ophv119/scratch/miniconda3/envs/llava/targets/x86_64-linux/lib/libcurand.so.10.3.6.82 $CONDA_PREFIX/lib64/libcurand.so
ln -sf /home/users/nus/ophv119/scratch/miniconda3/envs/llava/targets/x86_64-linux/lib/libcurand.so.10.3.6.82 $CONDA_PREFIX/lib64/libcurand.so.10

# 设置更全面的库路径环境变量
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/curand/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/curand/lib:$LIBRARY_PATH

# 设置编译器标志 - 关键是这些
export CFLAGS="-I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib64 -L$CONDA_PREFIX/targets/x86_64-linux/lib"
export CXXFLAGS="-I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib64 -L$CONDA_PREFIX/targets/x86_64-linux/lib"
export LDFLAGS="-L$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib64 -L$CONDA_PREFIX/targets/x86_64-linux/lib"

# 专门为 PyTorch C++ 扩展设置的环境变量
export TORCH_CUDA_ARCH_LIST=""
export FORCE_CUDA=1
export CUDA_HOME=$CONDA_PREFIX
export CUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX

# 设置其他环境变量
export PYTHONPATH=/home/users/nus/ophv119/LLaVA:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

export TRANSFORMERS_CACHE=/scratch/users/nus/ophv119/llava_model/
export HF_HOME=/scratch/users/nus/ophv119/llava_model/
export HF_HUB_CACHE=/scratch/users/nus/ophv119/llava_model/

export DS_SKIP_CUDA_CHECK=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments

# 清理之前失败的编译缓存
rm -rf /home/users/nus/ophv119/.cache/torch_extensions/

mkdir -p checkpoints/llava-v1.5-13b
mkdir -p checkpoints/llava-v1.5-13b-pretrain
mkdir -p checkpoints/llava-v1.5-13b-sdc-lora

deepspeed llava/train/train_xformers.py \
    --lora_enable True --lora_r 64 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.5-13b \
    --version v1 \
    --data_path /home/users/nus/ophv119/LLaVA/playground/data/diffusiondb_dataset/llava_finetune_data.json \
    --image_folder ./playground/data/diffusiondb_dataset \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-13b/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-13b-sdc-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb 2>&1 | tee logs/finetune_lora.log