# -*- coding: utf-8 -*-
"""
Skip-gram词嵌入模型 - 使用无监督学习方式训练词向量
"""

import os
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional
from collections import Counter
import random


class SkipGramModel:
    """Skip-gram词嵌入模型
    
    通过预测上下文词来学习词向量，不需要标签数据
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 64, 
                 window_size: int = 2, num_negative: int = 5):
        """初始化模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 嵌入向量维度
            window_size: 上下文窗口大小
            num_negative: 负采样数量
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_negative = num_negative
        self.model: Optional[tf.keras.Model] = None
        self.target_embedding: Optional[tf.keras.layers.Embedding] = None
        self.context_embedding: Optional[tf.keras.layers.Embedding] = None
        
    def build_model(self, pretrained_weights: np.ndarray = None) -> tf.keras.Model:
        """构建Skip-gram模型
        
        Args:
            pretrained_weights: 预训练权重（可选）
        
        Returns:
            构建好的Keras模型
        """
        # 目标词输入
        target_input = tf.keras.layers.Input(shape=(1,), name='target_input')
        # 上下文词输入
        context_input = tf.keras.layers.Input(shape=(1,), name='context_input')
        
        # 目标词嵌入层
        if pretrained_weights is not None:
            self.target_embedding = tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                weights=[pretrained_weights],
                trainable=True,
                name='target_embedding'
            )
        else:
            self.target_embedding = tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                embeddings_initializer='uniform',
                name='target_embedding'
            )
        
        # 上下文词嵌入层
        self.context_embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            embeddings_initializer='uniform',
            name='context_embedding'
        )
        
        # 获取嵌入向量
        target_vec = self.target_embedding(target_input)  # (batch, 1, dim)
        context_vec = self.context_embedding(context_input)  # (batch, 1, dim)
        
        # 点积计算相似度
        dot_product = tf.keras.layers.Dot(axes=-1, normalize=False)([target_vec, context_vec])
        dot_product = tf.keras.layers.Flatten()(dot_product)
        
        # Sigmoid输出
        output = tf.keras.layers.Activation('sigmoid', name='output')(dot_product)
        
        self.model = tf.keras.Model(
            inputs=[target_input, context_input],
            outputs=output
        )
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001) -> None:
        """编译模型"""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def generate_training_data(self, sequences: List[List[int]], 
                               word_counts: Counter) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成训练数据（正样本和负样本）
        
        Args:
            sequences: 编码后的句子列表
            word_counts: 词频统计
            
        Returns:
            (target_words, context_words, labels)
        """
        targets = []
        contexts = []
        labels = []
        
        # 计算采样概率（用于负采样）
        total_count = sum(word_counts.values())
        sampling_probs = np.zeros(self.vocab_size)
        for word_idx, count in word_counts.items():
            # 使用3/4次方来平滑分布
            sampling_probs[word_idx] = (count / total_count) ** 0.75
        sampling_probs /= sampling_probs.sum()
        
        for sequence in sequences:
            for i, target_word in enumerate(sequence):
                if target_word == 0:  # 跳过PAD
                    continue
                    
                # 正样本：窗口内的上下文词
                window_start = max(0, i - self.window_size)
                window_end = min(len(sequence), i + self.window_size + 1)
                
                for j in range(window_start, window_end):
                    if i != j and sequence[j] != 0:  # 跳过自己和PAD
                        targets.append(target_word)
                        contexts.append(sequence[j])
                        labels.append(1)
                        
                        # 负样本
                        for _ in range(self.num_negative):
                            neg_word = np.random.choice(self.vocab_size, p=sampling_probs)
                            while neg_word == target_word or neg_word == 0:
                                neg_word = np.random.choice(self.vocab_size, p=sampling_probs)
                            targets.append(target_word)
                            contexts.append(neg_word)
                            labels.append(0)
        
        return np.array(targets), np.array(contexts), np.array(labels)
    
    def get_embedding_weights(self) -> np.ndarray:
        """获取词嵌入权重"""
        if self.target_embedding is None:
            raise ValueError("模型尚未构建")
        return self.target_embedding.get_weights()[0]
    
    def save(self, path: str) -> None:
        """保存模型"""
        if self.model is None:
            raise ValueError("模型尚未构建")
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        model_path = path + '.keras' if not path.endswith('.keras') else path
        self.model.save(model_path)
        
        # 保存配置
        config = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'window_size': self.window_size,
            'num_negative': self.num_negative
        }
        import json
        config_path = path.rstrip('/') + '_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        print(f"模型已保存到 {model_path}")
    
    def load(self, path: str) -> None:
        """加载模型"""
        config_path = path.rstrip('/') + '_config.json'
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.vocab_size = config['vocab_size']
            self.embedding_dim = config['embedding_dim']
            self.window_size = config.get('window_size', 2)
            self.num_negative = config.get('num_negative', 5)
        
        model_path = path + '.keras' if not path.endswith('.keras') else path
        self.model = tf.keras.models.load_model(model_path)
        self.target_embedding = self.model.get_layer('target_embedding')
        self.context_embedding = self.model.get_layer('context_embedding')
        print(f"模型已从 {model_path} 加载")


def train_skipgram(data_path: str = None,
                   model_save_path: str = "models/skipgram_model",
                   vocab_save_path: str = "models/vocabulary.json",
                   embedding_dim: int = 64,
                   window_size: int = 3,
                   num_negative: int = 5,
                   epochs: int = 30,
                   batch_size: int = 256,
                   learning_rate: float = 0.001,
                   use_pretrained: bool = True) -> Tuple['SkipGramModel', 'Vocabulary']:
    """训练Skip-gram词嵌入模型"""
    from dataset import CampusDataset
    from preprocessor import ChineseTokenizer, Vocabulary
    from pretrained_embeddings import generate_campus_embeddings
    from collections import Counter
    
    # 加载数据
    dataset = CampusDataset(data_path)
    if data_path and os.path.exists(data_path):
        dataset.load()
    else:
        dataset.generate_from_previous_projects(
            xml_path="../1_University_Vocabulary_and_XML_Tools/university.xml",
            rdf_path="../2_RDF_Vocabulary_and_SPARQL_Tool/cs_data.ttl",
            ontology_path="../3_Ontology_Modeling_and_Reasoning/campus_instances.ttl"
        )
        if data_path:
            dataset.save(data_path)
    
    sentences = dataset.get_sentences()
    print(f"数据集大小: {len(sentences)} 条句子")
    
    # 分词和构建词汇表
    tokenizer = ChineseTokenizer()
    vocabulary = Vocabulary()
    
    tokenized_sentences = [tokenizer.tokenize(s) for s in sentences]
    vocabulary.build_vocab(tokenized_sentences, min_freq=1)
    print(f"词汇表大小: {len(vocabulary)}")
    
    # 编码句子
    encoded_sentences = [vocabulary.encode(tokens) for tokens in tokenized_sentences]
    
    # 统计词频
    word_counts = Counter()
    for seq in encoded_sentences:
        word_counts.update(seq)
    
    # 创建模型
    model = SkipGramModel(
        vocab_size=len(vocabulary),
        embedding_dim=embedding_dim,
        window_size=window_size,
        num_negative=num_negative
    )
    
    # 使用预训练权重
    pretrained_weights = None
    if use_pretrained:
        print("\n正在生成语义增强的词向量...")
        pretrained_weights = generate_campus_embeddings(vocabulary, embedding_dim)
    
    model.build_model(pretrained_weights)
    model.compile_model(learning_rate=learning_rate)
    
    # 生成训练数据
    print("\n正在生成训练数据...")
    targets, contexts, labels = model.generate_training_data(encoded_sentences, word_counts)
    print(f"训练样本数: {len(targets)} (正样本: {labels.sum()}, 负样本: {len(labels) - labels.sum()})")
    
    # 打乱数据
    indices = np.random.permutation(len(targets))
    targets = targets[indices]
    contexts = contexts[indices]
    labels = labels[indices]
    
    # 训练
    print(f"\n开始训练...")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {batch_size}")
    
    history = model.model.fit(
        [targets, contexts], labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    
    print("\n训练完成!")
    
    # 保存
    model.save(model_save_path)
    vocabulary.save(vocab_save_path)
    
    return model, vocabulary


if __name__ == "__main__":
    model, vocab = train_skipgram(
        data_path="data/campus_dataset.json",
        embedding_dim=64,
        epochs=30
    )
