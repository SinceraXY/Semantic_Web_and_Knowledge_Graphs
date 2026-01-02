# -*- coding: utf-8 -*-
"""
预处理模块 - 中文分词和词汇表管理
"""

import json
import os
import re
import string
from typing import List, Dict, Optional
from collections import Counter

import jieba


class ChineseTokenizer:
    """中文分词器"""
    
    # 中文标点符号
    CHINESE_PUNCTUATION = '，。！？、；：""''（）【】《》…—～·'
    
    def __init__(self):
        """初始化分词器"""
        # 加载jieba分词器
        jieba.initialize()
        
    def tokenize(self, text: str) -> List[str]:
        """对文本进行分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词后的词语列表
        """
        if not text or not text.strip():
            return []
        
        # 先移除标点符号
        clean_text = self.remove_punctuation(text)
        
        # 使用jieba进行分词
        tokens = list(jieba.cut(clean_text))
        
        # 过滤空白词语
        tokens = [t.strip() for t in tokens if t.strip()]
        
        return tokens
    
    def remove_punctuation(self, text: str) -> str:
        """移除标点符号和特殊字符
        
        Args:
            text: 输入文本
            
        Returns:
            移除标点后的文本
        """
        if not text:
            return ""
        
        # 移除英文标点
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 移除中文标点
        for punct in self.CHINESE_PUNCTUATION:
            text = text.replace(punct, ' ')
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class Vocabulary:
    """词汇表"""
    
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    PAD_IDX = 0
    UNK_IDX = 1
    
    def __init__(self):
        """初始化词汇表"""
        self.word2idx: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX
        }
        self.idx2word: Dict[int, str] = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN
        }
        self._next_idx = 2
        
    def build_vocab(self, sentences: List[List[str]], min_freq: int = 1) -> None:
        """从分词后的句子构建词汇表
        
        Args:
            sentences: 分词后的句子列表，每个句子是词语列表
            min_freq: 最小词频，低于此频率的词语将被忽略
        """
        # 统计词频
        word_freq = Counter()
        for sentence in sentences:
            word_freq.update(sentence)
        
        # 添加满足最小词频的词语
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = self._next_idx
                self.idx2word[self._next_idx] = word
                self._next_idx += 1
    
    def add_word(self, word: str) -> int:
        """添加单个词语到词汇表
        
        Args:
            word: 词语
            
        Returns:
            词语的索引
        """
        if word not in self.word2idx:
            self.word2idx[word] = self._next_idx
            self.idx2word[self._next_idx] = word
            self._next_idx += 1
        return self.word2idx[word]
    
    def encode(self, tokens: List[str]) -> List[int]:
        """将词语序列编码为整数序列
        
        Args:
            tokens: 词语列表
            
        Returns:
            整数索引列表
        """
        return [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        """将整数序列解码为词语序列
        
        Args:
            indices: 整数索引列表
            
        Returns:
            词语列表
        """
        return [self.idx2word.get(i, self.UNK_TOKEN) for i in indices]
    
    def get_word(self, idx: int) -> str:
        """根据索引获取词语
        
        Args:
            idx: 词语索引
            
        Returns:
            词语
        """
        return self.idx2word.get(idx, self.UNK_TOKEN)
    
    def get_index(self, word: str) -> int:
        """根据词语获取索引
        
        Args:
            word: 词语
            
        Returns:
            词语索引
        """
        return self.word2idx.get(word, self.UNK_IDX)
    
    def contains(self, word: str) -> bool:
        """检查词语是否在词汇表中
        
        Args:
            word: 词语
            
        Returns:
            是否存在
        """
        return word in self.word2idx
    
    def save(self, path: str) -> None:
        """保存词汇表到文件
        
        Args:
            path: 文件路径
        """
        data = {
            'word2idx': self.word2idx,
            'idx2word': {str(k): v for k, v in self.idx2word.items()}
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"词汇表已保存到 {path}，共 {len(self)} 个词语")
    
    def load(self, path: str) -> None:
        """从文件加载词汇表
        
        Args:
            path: 文件路径
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.word2idx = data['word2idx']
        self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        self._next_idx = max(self.idx2word.keys()) + 1
        
        print(f"词汇表已从 {path} 加载，共 {len(self)} 个词语")
    
    def get_all_words(self) -> List[str]:
        """获取所有词语（不包括特殊标记）
        
        Returns:
            词语列表
        """
        return [w for w in self.word2idx.keys() 
                if w not in (self.PAD_TOKEN, self.UNK_TOKEN)]
    
    def __len__(self) -> int:
        """返回词汇表大小"""
        return len(self.word2idx)
    
    def __contains__(self, word: str) -> bool:
        """检查词语是否在词汇表中"""
        return word in self.word2idx


if __name__ == "__main__":
    # 测试分词器
    tokenizer = ChineseTokenizer()
    
    test_texts = [
        "计算机科学系开设了数据结构课程",
        "王老师是计算机系的教授，他教授机器学习课程。",
        "学生们在图书馆自习！",
    ]
    
    print("分词测试:")
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        print(f"  原文: {text}")
        print(f"  分词: {tokens}")
        print()
    
    # 测试词汇表
    print("\n词汇表测试:")
    vocab = Vocabulary()
    
    # 构建词汇表
    tokenized_sentences = [tokenizer.tokenize(text) for text in test_texts]
    vocab.build_vocab(tokenized_sentences)
    
    print(f"词汇表大小: {len(vocab)}")
    print(f"词汇表内容: {list(vocab.word2idx.keys())[:20]}...")
    
    # 测试编码解码
    test_tokens = tokenized_sentences[0]
    encoded = vocab.encode(test_tokens)
    decoded = vocab.decode(encoded)
    
    print(f"\n编码解码测试:")
    print(f"  原始词语: {test_tokens}")
    print(f"  编码结果: {encoded}")
    print(f"  解码结果: {decoded}")
    
    # 测试未知词
    unknown_tokens = ["未知词", "测试"]
    encoded_unknown = vocab.encode(unknown_tokens)
    print(f"\n未知词测试:")
    print(f"  未知词语: {unknown_tokens}")
    print(f"  编码结果: {encoded_unknown}")
