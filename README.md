# Transformer-learn
Learn how to build a basics transformer with translation function

Referenced papers "Attention is All You Need " and links https://zhuanlan.zhihu.com/p/403433120

数据准备：将中英文对齐的句子转换成词汇索引，构造训练数据集。
Transformer 架构：包括 编码器（Encoder） 和 解码器（Decoder），分别用于处理源语言和目标语言。
训练 Transformer：使用 交叉熵损失函数 进行训练，并优化参数以提高翻译效果。
模型推理（测试）：实现 贪心解码（Greedy Decoding），让用户输入中文句子并得到英文翻译。

Data preparation: Convert the aligned Chinese and English sentences into vocabulary indices and construct a training dataset.
Transformer architecture: Includes encoder and decoder, which are used to process the source language and target language respectively.
Training Transformer: Use the cross entropy loss function for training and optimize the parameters to improve the translation effect.
Model inference (testing): Implement Greedy Decoding to allow users to input Chinese sentences and get English translations.
