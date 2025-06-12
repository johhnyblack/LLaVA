# 简单测试脚本
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

model_path = "/home/users/nus/ophv119/LLaVA/checkpoints/llava-v1.5-13b-sdc-lora"

try:
    # 尝试加载模型配置
    processor = LlavaNextProcessor.from_pretrained(model_path)
    print("✅ Processor 加载成功")
    
    # 尝试加载模型（使用CPU避免GPU内存问题）
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    print("✅ 模型加载成功！合并完成！")
    
except Exception as e:
    print(f"❌ 加载失败: {e}")