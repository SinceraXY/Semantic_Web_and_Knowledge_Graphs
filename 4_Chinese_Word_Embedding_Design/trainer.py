# -*- coding: utf-8 -*-
"""
训练模块 - 模型训练流程
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional

from model import EmbeddingModel
from preprocessor import ChineseTokenizer, Vocabulary


class Trainer:
    """模型训练器"""
    
    def __init__(self, model: EmbeddingModel, vocabulary: Vocabulary):
        """初始化训练器
        
        Args:
            model: 词嵌入模型
            vocabulary: 词汇表
        """
        self.model = model
        self.vocabulary = vocabulary
        self.history: Optional[tf.keras.callbacks.History] = None
        
    def prepare_data(self, sentences: List[str], labels: List[int],
                     tokenizer: ChineseTokenizer) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据
        
        Args:
            sentences: 句子列表
            labels: 标签列表
            tokenizer: 分词器
            
        Returns:
            (X, y) 训练数据元组
        """
        if len(sentences) != len(labels):
            raise ValueError("句子数量和标签数量不匹配")
        
        if len(sentences) == 0:
            raise ValueError("训练数据为空")
        
        # 分词
        tokenized_sentences = [tokenizer.tokenize(s) for s in sentences]
        
        # 编码
        encoded_sentences = [self.vocabulary.encode(tokens) for tokens in tokenized_sentences]
        
        # 填充/截断到固定长度
        X = tf.keras.preprocessing.sequence.pad_sequences(
            encoded_sentences,
            maxlen=self.model.max_length,
            padding='post',
            truncating='post',
            value=self.vocabulary.PAD_IDX
        )
        
        y = np.array(labels)
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 10, batch_size: int = 32,
              validation_split: float = 0.2,
              verbose: int = 1) -> tf.keras.callbacks.History:
        """训练模型
        
        Args:
            X: 输入数据
            y: 标签数据
            epochs: 训练轮数
            batch_size: 批次大小
            validation_split: 验证集比例
            verbose: 日志详细程度
            
        Returns:
            训练历史
        """
        if self.model.model is None:
            self.model.build_model()
            self.model.compile_model()
        
        print(f"\n开始训练...")
        print(f"  训练样本数: {len(X)}")
        print(f"  训练轮数: {epochs}")
        print(f"  批次大小: {batch_size}")
        print(f"  验证集比例: {validation_split}")
        
        # 添加早停回调
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # 训练模型
        self.history = self.model.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        print("\n训练完成!")
        
        return self.history
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型
        
        Args:
            X: 输入数据
            y: 标签数据
            
        Returns:
            评估指标字典
        """
        if self.model.model is None:
            raise ValueError("模型尚未训练")
        
        loss, accuracy = self.model.model.evaluate(X, y, verbose=0)
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测
        
        Args:
            X: 输入数据
            
        Returns:
            预测结果
        """
        if self.model.model is None:
            raise ValueError("模型尚未训练")
        
        return self.model.model.predict(X, verbose=0)
    
    def get_training_history(self) -> Optional[Dict]:
        """获取训练历史
        
        Returns:
            训练历史字典
        """
        if self.history is None:
            return None
        return self.history.history


def train_word_embedding(data_path: str = None,
                         model_save_path: str = "models/embedding_model",
                         vocab_save_path: str = "models/vocabulary.json",
                         embedding_dim: int = 64,
                         max_length: int = 50,
                         epochs: int = 20,
                         batch_size: int = 32,
                         learning_rate: float = 0.001,
                         use_pretrained: bool = True,
                         trainable_embedding: bool = True) -> Tuple[EmbeddingModel, Vocabulary]:
    """训练词嵌入模型的便捷函数
    
    Args:
        data_path: 数据集路径
        model_save_path: 模型保存路径
        vocab_save_path: 词汇表保存路径
        embedding_dim: 嵌入维度
        max_length: 序列最大长度
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        use_pretrained: 是否使用预训练词向量
        trainable_embedding: 嵌入层是否可训练
        
    Returns:
        (模型, 词汇表) 元组
    """
    from dataset import CampusDataset
    from pretrained_embeddings import generate_campus_embeddings
    
    # 加载或生成数据集
    dataset = CampusDataset(data_path)
    
    if data_path and os.path.exists(data_path):
        dataset.load()
    else:
        # 生成数据集
        dataset.generate_from_previous_projects(
            xml_path="../1_University_Vocabulary_and_XML_Tools/university.xml",
            rdf_path="../2_RDF_Vocabulary_and_SPARQL_Tool/cs_data.ttl",
            ontology_path="../3_Ontology_Modeling_and_Reasoning/campus_instances.ttl"
        )
        if data_path:
            dataset.save(data_path)
    
    sentences = dataset.get_sentences()
    labels = dataset.get_labels()
    
    print(f"数据集大小: {len(sentences)} 条句子")
    
    # 初始化分词器和词汇表
    tokenizer = ChineseTokenizer()
    vocabulary = Vocabulary()
    
    # 分词并构建词汇表
    tokenized_sentences = [tokenizer.tokenize(s) for s in sentences]
    vocabulary.build_vocab(tokenized_sentences, min_freq=1)
    
    print(f"词汇表大小: {len(vocabulary)}")
    
    # 创建模型
    model = EmbeddingModel(
        vocab_size=len(vocabulary),
        embedding_dim=embedding_dim,
        max_length=max_length,
        num_classes=len(dataset.LABELS)
    )
    
    # 使用预训练词向量
    if use_pretrained:
        print("\n正在生成语义增强的词向量...")
        pretrained_weights = generate_campus_embeddings(vocabulary, embedding_dim)
        model.set_pretrained_weights(pretrained_weights)
    
    model.build_model(trainable_embedding=trainable_embedding)
    model.compile_model(learning_rate=learning_rate)
    
    # 创建训练器
    trainer = Trainer(model, vocabulary)
    
    # 准备数据
    X, y = trainer.prepare_data(sentences, labels, tokenizer)
    
    # 训练
    trainer.train(X, y, epochs=epochs, batch_size=batch_size)
    
    # 评估
    metrics = trainer.evaluate(X, y)
    print(f"\n最终评估结果:")
    print(f"  损失: {metrics['loss']:.4f}")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    
    # 保存模型和词汇表
    model.save(model_save_path)
    vocabulary.save(vocab_save_path)
    
    return model, vocabulary


if __name__ == "__main__":
    # 训练词嵌入模型
    model, vocabulary = train_word_embedding(
        data_path="data/campus_dataset.json",
        model_save_path="models/embedding_model",
        vocab_save_path="models/vocabulary.json",
        embedding_dim=64,
        epochs=20,
        batch_size=16
    )
