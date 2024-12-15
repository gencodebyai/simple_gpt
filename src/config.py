class GPTConfig:
    def __init__(self):
        self.vocab_size = 50257  # GPT-2默认词表大小
        self.block_size = 1024   # 上下文窗口大小
        self.n_layer = 6         # Transformer层数
        self.n_head = 8          # 注意力头数
        self.n_embd = 512        # 嵌入维度
        self.dropout = 0.1       # Dropout比率
        self.learning_rate = 3e-4
        self.max_epochs = 10
        self.batch_size = 32
        self.device = 'cuda'     # 或'cpu' 