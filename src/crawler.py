import requests
from bs4 import BeautifulSoup
import os
import time
import random
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class WebCrawler:
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        os.makedirs(output_dir, exist_ok=True)
        
    def crawl_wiki(self, num_pages=1000):
        """爬取中文维基百科文章"""
        base_url = "https://zh.wikipedia.org/wiki/Special:Random"
        texts = []
        
        for i in tqdm(range(num_pages), desc="爬取维基百科"):
            try:
                response = requests.get(base_url, headers=self.headers)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 获取正文内容
                content = soup.find(id="mw-content-text")
                if content:
                    paragraphs = content.find_all('p')
                    text = '\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                    if text:
                        texts.append(text)
                
                time.sleep(random.uniform(1, 3))  # 随机延迟，避免被封
            except Exception as e:
                logging.error(f"爬取出错: {str(e)}")
                continue
        
        # 保存原始数据
        output_file = os.path.join(self.output_dir, 'wiki_articles.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(texts))
        
        return output_file

    def crawl_news(self, num_pages=1000):
        """爬取新闻网站"""
        # 这里可以实现新闻网站的爬取逻辑
        pass 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_pages', type=int, default=1000, help='要爬取的页面数量')
    parser.add_argument('--output_dir', type=str, default='data/raw', help='输出目录')
    args = parser.parse_args()
    
    crawler = WebCrawler(output_dir=args.output_dir)
    crawler.crawl_wiki(num_pages=args.num_pages)