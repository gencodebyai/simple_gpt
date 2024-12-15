# 处理后数据说明

此目录存放经过预处理的数据文件。

## 数据格式

1. 文件命名：train.txt, valid.txt, test.txt
2. 文件编码：UTF-8
3. 数据格式：每行为一个已经过分词和编码的样本

## 预处理步骤

数据经过以下处理：
1. 文本清理（去除特殊字符、统一格式等）
2. 分词
3. 编码（转换为词表索引） 