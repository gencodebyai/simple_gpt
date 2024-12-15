import os
import numpy as np
from sklearn.model_selection import train_test_split
import jieba
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TextPreprocessor:
    def __init__(self, raw_dir='data/raw', processed_dir='data/processed'):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        
    def clean_text(self, text):
        """清理文本"""
        # 去除特殊字符
        text = ''.join([char for char in text if '\u4e00' <= char <= '\u9fff' or char in '，。！？'])
        return text
    
    def tokenize(self, text):
        """分词"""
        return list(jieba.cut(text))
    
    def build_vocab(self, texts, max_vocab_size=50000):
        """构建词表"""
        word_freq = {}
        for text in texts:
            for word in self.tokenize(text):
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 选择最常见的词构建词表
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in sorted_words[:max_vocab_size-2]:
            vocab[word] = len(vocab)
        
        return vocab
    
    def process_data(self, test_size=0.1, val_size=0.1):
        """处理数据并划分训练集、验证集和测试集"""
        # 读取原始数据
        raw_files = [f for f in os.listdir(self.raw_dir) if f.endswith('.txt')]
        texts = []
        for file in raw_files:
            with open(os.path.join(self.raw_dir, file), 'r', encoding='utf-8') as f:
                texts.extend(f.read().split('\n'))
        
        # 清理文本
        texts = [self.clean_text(text) for text in tqdm(texts, desc="清理文本")]
        texts = [text for text in texts if len(text) > 10]  # 过滤太短的文本
        
        # 构建词表
        vocab = self.build_vocab(texts)
        
        # 保存词表
        with open(os.path.join(self.processed_dir, 'vocab.txt'), 'w', encoding='utf-8') as f:
            for word, idx in vocab.items():
                f.write(f'{word}\t{idx}\n')
        
        # 划分数据集
        train_texts, test_texts = train_test_split(texts, test_size=test_size, random_state=42)
        train_texts, val_texts = train_test_split(train_texts, test_size=val_size/(1-test_size), random_state=42)
        
        # 保存处理后的数据
        datasets = {
            'train.txt': train_texts,
            'valid.txt': val_texts,
            'test.txt': test_texts
        }
        
        for filename, data in datasets.items():
            output_path = os.path.join(self.processed_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(data))
            logging.info(f"保存{filename}，共{len(data)}条数据") 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--raw_dir', type=str, default='data/raw', help='原始数据目录')
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='处理后数据目录')
    args = parser.parse_args()
    
    preprocessor = TextPreprocessor(raw_dir=args.raw_dir, processed_dir=args.processed_dir)
    preprocessor.process_data(test_size=args.test_size, val_size=args.val_size)