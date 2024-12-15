import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
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

def get_device():
    """自动检测并返回可用的设备（GPU/CPU）"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # 支持 Mac M1/M2 的 MPS 加速
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train(config, model, train_dataset):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # 添加梯度累积步数
    gradient_accumulation_steps = 4  # 相当于模拟 batch_size * 4
    
    for epoch in range(config.max_epochs):
        total_loss = 0
        optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.max_epochs}')
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(config.device), y.to(config.device)
            
            # 前向传播
            logits, loss = model(x, y)
            
            # 缩放损失以适应梯度累积
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # 每 gradient_accumulation_steps 步进行一次参数更新
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')
        
        # 保存模型
        if (epoch + 1) % 5 == 0:  # 每5个epoch保存一次
            save_dir = 'checkpoints'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{save_dir}/model_epoch_{epoch+1}.pt')

    # 保存最终模型
    torch.save(model.state_dict(), 'checkpoints/model_final.pt')

def parse_args():
    parser = argparse.ArgumentParser(description='GPT 模型训练脚本')
    parser.add_argument('--data_path', type=str, required=True, help='训练数据路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='训练批次大小')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--block_size', type=int, default=128, help='序列长度')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建配置对象
    config = ModelConfig()
    config.max_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.block_size = args.block_size
    
    # 设置设备
    device = get_device()
    print(f"Using device: {device}")
    config.device = device
    
    # 加载数据集
    train_dataset = TextDataset(args.data_path, block_size=args.block_size)
    
    # 更新配置中的词表大小
    config.vocab_size = train_dataset.get_vocab_size()
    
    # 创建模型
    model = GPT(config).to(device)
    
    # 开始训练
    train(config, model, train_dataset)

if __name__ == '__main__':
    main() 