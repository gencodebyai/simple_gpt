import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data_path, block_size):
        # 读取文本文件
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 创建字符级别的词表
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # 创建编码解码字典
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
        # 编码整个文本
        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        self.block_size = block_size

    def encode(self, text):
        """将文本转换为数字序列"""
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        """将数字序列转换回文本"""
        return ''.join([self.itos[i] for i in ids])

    def __len__(self):
        """返回数据集中可能的序列数量"""
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        """获取训练样本"""
        # 确保索引有效
        if idx < 0 or idx >= len(self):
            raise IndexError("索引超出范围")
            
        # 获取输入序列和目标序列
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

    def get_vocab_size(self):
        """返回词表大小"""
        return self.vocab_size