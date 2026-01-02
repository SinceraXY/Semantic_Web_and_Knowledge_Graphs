# -*- coding: utf-8 -*-
"""
模型模块 - 基于TensorFlow的词嵌入模型
支持预训练词向量初始化
"""

import os
import numpy as np
import tensorflow as tf
from typing import Optional


class EmbeddingModel:
    """词嵌入模型
    
    基于TensorFlow实现，使用文本分类任务学习词嵌入。
    支持预训练词向量初始化。
    模型架构: Embedding -> GlobalAveragePooling1D -> Dense -> Dense(softmax)
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 64, 
                 max_length: int = 50, num_classes: int = 7):
        """初始化模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 嵌入向量维度
            max_length: 序列最大长度
            num_classes: 分类类别数
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_classes = num_classes
        self.model: Optional[tf.keras.Model] = None
        self._pretrained_weights: Optional[np.ndarray] = None
        
    def set_pretrained_weights(self, weights: np.ndarray) -> None:
        """设置预训练词向量权重
        
        Args:
            weights: 预训练权重矩阵，形状为 (vocab_size, embedding_dim)
        """
        if weights.shape[0] != self.vocab_size:
            raise ValueError(f"权重矩阵行数 {weights.shape[0]} 与词汇表大小 {self.vocab_size} 不匹配")
        self._pretrained_weights = weights
        self.embedding_dim = weights.shape[1]
        print(f"已设置预训练词向量，维度: {self.embedding_dim}")
        
    def build_model(self, trainable_embedding: bool = True) -> tf.keras.Model:
        """构建模型架构
        
        Args:
            trainable_embedding: 嵌入层是否可训练
        
        Returns:
            构建好的Keras模型
        """
        # 创建嵌入层
        if self._pretrained_weights is not None:
            # 使用预训练权重初始化
            embedding_layer = tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                weights=[self._pretrained_weights],
                trainable=trainable_embedding,
                name='embedding'
            )
            print(f"使用预训练词向量初始化嵌入层 (trainable={trainable_embedding})")
        else:
            # 随机初始化
            embedding_layer = tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                name='embedding'
            )
        
        self.model = tf.keras.Sequential([
            embedding_layer,
            # 全局平均池化：将变长序列转换为固定长度向量
            tf.keras.layers.GlobalAveragePooling1D(name='pooling'),
            # 简化的全连接层 - 减少过拟合
            tf.keras.layers.Dense(32, activation='relu', 
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                  name='dense1'),
            tf.keras.layers.Dropout(0.5, name='dropout1'),
            # 输出层
            tf.keras.layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001) -> None:
        """编译模型
        
        Args:
            learning_rate: 学习率
        """
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_embedding_layer(self) -> tf.keras.layers.Embedding:
        """获取嵌入层
        
        Returns:
            嵌入层
        """
        if self.model is None:
            raise ValueError("模型尚未构建，请先调用 build_model()")
        
        return self.model.get_layer('embedding')
    
    def get_embedding_weights(self) -> np.ndarray:
        """获取嵌入层权重
        
        Returns:
            嵌入权重矩阵，形状为 (vocab_size, embedding_dim)
        """
        embedding_layer = self.get_embedding_layer()
        return embedding_layer.get_weights()[0]
    
    def get_word_embedding(self, word_index: int) -> np.ndarray:
        """获取指定词语的嵌入向量
        
        Args:
            word_index: 词语在词汇表中的索引
            
        Returns:
            嵌入向量
        """
        weights = self.get_embedding_weights()
        
        if word_index < 0 or word_index >= self.vocab_size:
            raise ValueError(f"词语索引 {word_index} 超出范围 [0, {self.vocab_size})")
        
        return weights[word_index]
    
    def save(self, path: str) -> None:
        """保存模型
        
        Args:
            path: 保存路径（不含扩展名）
        """
        if self.model is None:
            raise ValueError("模型尚未构建，无法保存")
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Keras 3 需要 .keras 扩展名
        model_path = path + '.keras' if not path.endswith('.keras') else path
        self.model.save(model_path)
        
        # 同时保存模型配置
        config = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'num_classes': self.num_classes
        }
        
        config_path = path.rstrip('/') + '_config.json'
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        print(f"模型已保存到 {model_path}")
    
    def load(self, path: str) -> None:
        """加载模型
        
        Args:
            path: 模型路径（不含扩展名）
        """
        # 加载模型配置
        config_path = path.rstrip('/') + '_config.json'
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.vocab_size = config['vocab_size']
            self.embedding_dim = config['embedding_dim']
            self.max_length = config['max_length']
            self.num_classes = config['num_classes']
        
        # Keras 3 需要 .keras 扩展名
        model_path = path + '.keras' if not path.endswith('.keras') else path
        self.model = tf.keras.models.load_model(model_path)
        print(f"模型已从 {model_path} 加载")
    
    def summary(self) -> None:
        """打印模型摘要"""
        if self.model is None:
            raise ValueError("模型尚未构建")
        self.model.summary()


if __name__ == "__main__":
    # 测试模型
    print("测试词嵌入模型...")
    
    # 创建模型
    vocab_size = 1000
    embedding_dim = 64
    model = EmbeddingModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
    
    # 构建和编译模型
    model.build_model()
    model.compile_model()
    
    # 打印模型摘要
    print("\n模型架构:")
    model.summary()
    
    # 测试获取嵌入向量
    print(f"\n嵌入层权重形状: {model.get_embedding_weights().shape}")
    
    # 获取某个词的嵌入向量
    word_idx = 10
    embedding = model.get_word_embedding(word_idx)
    print(f"词语索引 {word_idx} 的嵌入向量形状: {embedding.shape}")
    print(f"嵌入向量前5个值: {embedding[:5]}")
