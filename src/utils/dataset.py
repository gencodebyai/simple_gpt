import torch
from torch.utils.data import Dataset
import os

class TextDataset(Dataset):
    def __init__(self, data_path, block_size):
        # 首先加载词表
        vocab_path = os.path.join(os.path.dirname(data_path), 'vocab.txt')
        self.load_vocab(vocab_path)
        
        # 读取文本文件
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 编码整个文本
        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        self.block_size = block_size

    def load_vocab(self, vocab_path):
        """加载词表文件"""
        try:
            self.chars = []
            self.stoi = {}
            self.itos = {}
            
            # 添加特殊token
            special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
            for i, token in enumerate(special_tokens):
                self.chars.append(token)
                self.stoi[token] = i
                self.itos[i] = token
            
            # 读取词表文件
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, start=len(special_tokens)):
                    token = line.strip().split('\t')[0]  # 假设格式为 "token\tindex"
                    self.chars.append(token)
                    self.stoi[token] = i
                    self.itos[i] = token
            
            self.vocab_size = len(self.chars)
            
            # 设置特殊token的索引
            self.pad_token_id = self.stoi['<PAD>']
            self.unk_token_id = self.stoi['<UNK>']
            self.bos_token_id = self.stoi['<BOS>']
            self.eos_token_id = self.stoi['<EOS>']
            
            print(f"Loaded vocabulary from {vocab_path}")
            print(f"Vocabulary size: {self.vocab_size}")
            
        except Exception as e:
            print(f"Error loading vocabulary: {e}")
            raise

    def encode(self, text):
        """将文本转换为数字序列，对于未知字符返回UNK token的id"""
        return [self.stoi.get(c, self.unk_token_id) for c in text]

    def decode(self, ids):
        """将数字序列转换回文本，跳过特殊token"""
        text = []
        for i in ids:
            if i >= len(self.itos):  # 处理超出词表范围的索引
                continue
            token = self.itos[i]
            if token not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                text.append(token)
        return ''.join(text)

    def __len__(self):
        """返回数据集中可能的序列数量"""
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        """获取训练样本"""
        if idx < 0 or idx >= len(self):
            raise IndexError("索引超出范围")
            
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

    def get_vocab_size(self):
        """返回词表大小"""
        return self.vocab_size

    def print_vocab_info(self):
        """打印词表信息，用于调试"""
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Special tokens: ['<PAD>', '<UNK>', '<BOS>', '<EOS>']")
        print(f"First 10 tokens in vocab: {self.chars[:10]}")