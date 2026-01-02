# -*- coding: utf-8 -*-
"""
词向量工具模块 - 词向量操作和相似度计算
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Union

from preprocessor import Vocabulary


class EmbeddingUtils:
    """词向量工具类"""
    
    def __init__(self, model, vocabulary: Vocabulary):
        """初始化工具类
        
        Args:
            model: 词嵌入模型（EmbeddingModel或SkipGramModel）
            vocabulary: 词汇表
        """
        self.model = model
        self.vocabulary = vocabulary
        self._embedding_cache: Optional[np.ndarray] = None
        
    def get_word_vector(self, word: str) -> np.ndarray:
        """获取词语的嵌入向量
        
        Args:
            word: 词语
            
        Returns:
            嵌入向量
        """
        # 获取词语索引
        idx = self.vocabulary.get_index(word)
        
        # 返回嵌入向量
        embeddings = self.get_all_embeddings()
        return embeddings[idx]
    
    def get_all_embeddings(self) -> np.ndarray:
        """获取所有词语的嵌入向量
        
        Returns:
            嵌入矩阵
        """
        if self._embedding_cache is None:
            self._embedding_cache = self.model.get_embedding_weights()
        return self._embedding_cache
    
    def cosine_similarity(self, word1: str, word2: str) -> float:
        """计算两个词语的余弦相似度
        
        Args:
            word1: 第一个词语
            word2: 第二个词语
            
        Returns:
            余弦相似度，范围[-1, 1]
        """
        vec1 = self.get_word_vector(word1)
        vec2 = self.get_word_vector(word2)
        
        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # 确保结果在[-1, 1]范围内（处理浮点数精度问题）
        return float(np.clip(similarity, -1.0, 1.0))
    
    def find_similar_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """找到最相似的top-k个词语
        
        Args:
            word: 目标词语
            top_k: 返回的相似词数量
            
        Returns:
            (词语, 相似度)列表，按相似度降序排列
        """
        # 获取目标词向量
        target_vec = self.get_word_vector(word)
        target_norm = np.linalg.norm(target_vec)
        
        if target_norm == 0:
            return []
        
        # 获取所有嵌入
        all_embeddings = self.get_all_embeddings()
        
        # 计算所有词语与目标词的相似度
        similarities = []
        
        for idx in range(len(self.vocabulary)):
            # 跳过特殊标记
            w = self.vocabulary.get_word(idx)
            if w in (Vocabulary.PAD_TOKEN, Vocabulary.UNK_TOKEN):
                continue
            
            # 跳过目标词本身
            if w == word:
                continue
            
            vec = all_embeddings[idx]
            vec_norm = np.linalg.norm(vec)
            
            if vec_norm == 0:
                continue
            
            sim = np.dot(target_vec, vec) / (target_norm * vec_norm)
            sim = float(np.clip(sim, -1.0, 1.0))
            similarities.append((w, sim))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def word_analogy(self, word_a: str, word_b: str, word_c: str, 
                     top_k: int = 5) -> List[Tuple[str, float]]:
        """词语类比：A之于B，如同C之于？
        
        计算: vec(B) - vec(A) + vec(C)
        
        Args:
            word_a: 词语A
            word_b: 词语B
            word_c: 词语C
            top_k: 返回的结果数量
            
        Returns:
            (词语, 相似度)列表
        """
        vec_a = self.get_word_vector(word_a)
        vec_b = self.get_word_vector(word_b)
        vec_c = self.get_word_vector(word_c)
        
        # 计算目标向量
        target_vec = vec_b - vec_a + vec_c
        target_norm = np.linalg.norm(target_vec)
        
        if target_norm == 0:
            return []
        
        # 获取所有嵌入
        all_embeddings = self.get_all_embeddings()
        
        # 计算相似度
        similarities = []
        exclude_words = {word_a, word_b, word_c, Vocabulary.PAD_TOKEN, Vocabulary.UNK_TOKEN}
        
        for idx in range(len(self.vocabulary)):
            w = self.vocabulary.get_word(idx)
            if w in exclude_words:
                continue
            
            vec = all_embeddings[idx]
            vec_norm = np.linalg.norm(vec)
            
            if vec_norm == 0:
                continue
            
            sim = np.dot(target_vec, vec) / (target_norm * vec_norm)
            sim = float(np.clip(sim, -1.0, 1.0))
            similarities.append((w, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def export_embeddings(self, output_dir: str) -> None:
        """导出嵌入向量为TSV格式（用于TensorFlow Projector）
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有嵌入
        all_embeddings = self.get_all_embeddings()
        
        # 导出向量文件
        vectors_path = os.path.join(output_dir, 'vectors.tsv')
        metadata_path = os.path.join(output_dir, 'metadata.tsv')
        
        with open(vectors_path, 'w', encoding='utf-8') as vec_file, \
             open(metadata_path, 'w', encoding='utf-8') as meta_file:
            
            # 注意：单列元数据不需要标题行
            for idx in range(len(self.vocabulary)):
                word = self.vocabulary.get_word(idx)
                
                # 跳过PAD标记
                if word == Vocabulary.PAD_TOKEN:
                    continue
                
                # 写入向量
                vec = all_embeddings[idx]
                vec_str = '\t'.join(map(str, vec))
                vec_file.write(vec_str + '\n')
                
                # 写入元数据（不需要标题行）
                meta_file.write(word + '\n')
        
        print(f"嵌入向量已导出到 {output_dir}")
        print(f"  向量文件: {vectors_path}")
        print(f"  元数据文件: {metadata_path}")
        print(f"\n可以上传到 https://projector.tensorflow.org/ 进行可视化")
    
    def get_vocabulary_words(self) -> List[str]:
        """获取词汇表中的所有词语（不包括特殊标记）
        
        Returns:
            词语列表
        """
        return self.vocabulary.get_all_words()


if __name__ == "__main__":
    # 测试词向量工具
    print("测试词向量工具...")
    
    # 这里需要先训练模型
    # 假设模型和词汇表已经存在
    try:
        from trainer import train_word_embedding
        
        # 训练模型
        model, vocabulary = train_word_embedding(
            data_path="data/campus_dataset.json",
            model_save_path="models/embedding_model",
            vocab_save_path="models/vocabulary.json",
            epochs=10
        )
        
        # 创建工具类
        utils = EmbeddingUtils(model, vocabulary)
        
        # 测试获取词向量
        test_words = ["计算机", "学生", "教授", "课程"]
        print("\n词向量测试:")
        for word in test_words:
            if vocabulary.contains(word):
                vec = utils.get_word_vector(word)
                print(f"  {word}: 向量维度={vec.shape}, 前3个值={vec[:3]}")
            else:
                print(f"  {word}: 不在词汇表中")
        
        # 测试相似度计算
        print("\n相似度测试:")
        word_pairs = [("学生", "教授"), ("课程", "学习"), ("计算机", "软件")]
        for w1, w2 in word_pairs:
            if vocabulary.contains(w1) and vocabulary.contains(w2):
                sim = utils.cosine_similarity(w1, w2)
                print(f"  {w1} - {w2}: {sim:.4f}")
        
        # 测试相似词查找
        print("\n相似词查找:")
        for word in ["学生", "课程"]:
            if vocabulary.contains(word):
                similar = utils.find_similar_words(word, top_k=5)
                print(f"  与'{word}'最相似的词:")
                for w, sim in similar:
                    print(f"    {w}: {sim:.4f}")
        
        # 导出嵌入
        utils.export_embeddings("output/embeddings")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
