class ModelConfig:
    def __init__(
        self,
        vocab_size=50257,
        block_size=128,
        n_embd=512,
        n_head=8,
        n_layer=6,
        dropout=0.1,
        learning_rate=3e-4,
        max_epochs=10,
        batch_size=32,
        device=None
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device 