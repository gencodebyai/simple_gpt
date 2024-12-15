import torch
import argparse
from model import GPT
from config import GPTConfig

def generate_text(model, prompt, max_length=100, temperature=0.8):
    model.eval()
    with torch.no_grad():
        # 将prompt转换为token indices
        # 这里需要实现tokenizer
        prompt_tokens = torch.tensor([[0]]) # 示例
        
        generated = model.generate(
            prompt_tokens,
            max_new_tokens=max_length,
            temperature=temperature
        )
        
        # 将生成的token转换回文本
        # 这里需要实现detokenizer
        return generated

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=100)
    args = parser.parse_args()
    
    config = GPTConfig()
    model = GPT(config)
    model.load_state_dict(torch.load(args.model_path))
    model.to(config.device)
    
    generated_text = generate_text(
        model,
        args.prompt,
        max_length=args.max_length
    )
    print(generated_text) 