import torch
import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.model.gpt import GPT
from src.utils.config import ModelConfig
from src.utils.dataset import TextDataset

def generate_text(model, prompt, dataset, max_length=100, temperature=0.8):
    """
    生成文本
    Args:
        model: GPT模型实例
        prompt: 输入提示文本
        dataset: 数据集实例，用于编码和解码
        max_length: 最大生成长度
        temperature: 采样温度
    Returns:
        str: 生成的文本
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 打印词表信息用于调试
    dataset.print_vocab_info()
    
    with torch.no_grad():
        try:
            # 将提示文本转换为token序列
            prompt_tokens = torch.tensor([dataset.encode(prompt)]).to(device)
            print(f"Encoded prompt: {prompt_tokens}")
            
            # 生成token序列
            generated_tokens = model.generate(
                prompt_tokens,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=50
            )
            
            # 将生成的token序列转换回文本
            generated_text = dataset.decode(generated_tokens[0].tolist())
            return prompt + generated_text
            
        except Exception as e:
            print(f"Error in text generation: {e}")
            return f"Error: Unable to generate text. {str(e)}"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt for text generation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data (for vocab)')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (0.0 to 1.0)')
    parser.add_argument('--block_size', type=int, default=128, help='Block size for the model')
    return parser.parse_args()

def load_model_and_adjust_weights(model_path, config, device):
    """
    加载模型并调整权重以匹配新的词表大小
    """
    # 创建新模型
    model = GPT(config)
    
    try:
        # 加载预训练权重
        state_dict = torch.load(model_path, map_location=device)
        
        # 获取预训练模型的词表大小
        old_vocab_size = state_dict['tok_emb.weight'].shape[0]
        new_vocab_size = config.vocab_size
        print(f"Adjusting vocabulary size from {old_vocab_size} to {new_vocab_size}")
        
        # 调整嵌入层权重
        if 'tok_emb.weight' in state_dict:
            old_embed = state_dict['tok_emb.weight']
            new_embed = torch.zeros((new_vocab_size, old_embed.shape[1]))
            # 复制较小维度的权重
            min_vocab_size = min(old_vocab_size, new_vocab_size)
            new_embed[:min_vocab_size] = old_embed[:min_vocab_size]
            state_dict['tok_emb.weight'] = new_embed
        
        # 调整输出层权重
        if 'head.weight' in state_dict:
            old_head = state_dict['head.weight']
            new_head = torch.zeros((new_vocab_size, old_head.shape[1]))
            # 复制较小维度的权重
            min_vocab_size = min(old_vocab_size, new_vocab_size)
            new_head[:min_vocab_size] = old_head[:min_vocab_size]
            state_dict['head.weight'] = new_head
        
        # 加载调整后的权重
        model.load_state_dict(state_dict)
        print("Successfully loaded and adjusted model weights")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    return model

def main():
    args = parse_args()
    
    # 加载数据集以获取词表
    dataset = TextDataset(args.data_path, block_size=args.block_size)
    
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型配置
    config = ModelConfig(
        vocab_size=dataset.get_vocab_size(),
        block_size=args.block_size
    )
    
    # 加载并调整模型
    model = load_model_and_adjust_weights(args.model_path, config, device)
    model = model.to(device)
    model.eval()
    
    # 生成文本
    generated_text = generate_text(
        model=model,
        prompt=args.prompt,
        dataset=dataset,
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated text:\n{generated_text}")

if __name__ == '__main__':
    main() 