# -*- coding: utf-8 -*-
"""
预训练词向量模块 - 加载和使用预训练中文词向量
使用语义类别初始化方法生成校园领域词向量
"""

import os
import numpy as np
from typing import Dict, Optional, Tuple, List


class PretrainedEmbeddings:
    """预训练词向量加载器"""
    
    def __init__(self, embedding_dim: int = 64):
        """初始化
        
        Args:
            embedding_dim: 词向量维度
        """
        self.embedding_dim = embedding_dim
        self.word_vectors: Optional[Dict[str, np.ndarray]] = None
        
    def load_from_dict(self, word_vectors: Dict[str, np.ndarray]) -> None:
        """从字典加载词向量
        
        Args:
            word_vectors: 词语到向量的映射字典
        """
        self.word_vectors = word_vectors
        if word_vectors:
            first_vec = next(iter(word_vectors.values()))
            self.embedding_dim = len(first_vec)
            
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """获取词语的向量
        
        Args:
            word: 词语
            
        Returns:
            词向量，如果不存在返回None
        """
        if self.word_vectors is not None:
            return self.word_vectors.get(word)
        return None
    
    def contains(self, word: str) -> bool:
        """检查词语是否在预训练词向量中"""
        if self.word_vectors is not None:
            return word in self.word_vectors
        return False


def generate_campus_embeddings(vocabulary, embedding_dim: int = 100) -> np.ndarray:
    """为校园领域词汇生成语义相关的词向量
    
    使用基于语义类别的初始化方法，让同类词语的向量更接近
    
    Args:
        vocabulary: Vocabulary对象
        embedding_dim: 嵌入维度
        
    Returns:
        嵌入矩阵
    """
    vocab_size = len(vocabulary)
    embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))
    
    # PAD向量保持为零
    embedding_matrix[vocabulary.PAD_IDX] = np.zeros(embedding_dim)
    
    # 定义语义类别及其关键词
    semantic_categories = {
        'course': ['课程', '数据结构', '线性代数', '操作系统', '机器学习', '深度学习',
                   '算法', '编程', '数据库', '网络', '软件', '编译', '人工智能',
                   '计算机', '程序', '学分', '必修', '选修', '实验', '设计'],
        'teacher': ['教授', '老师', '导师', '讲师', '教师', '副教授', '博导',
                    '指导', '教学', '授课', '批改', '辅导'],
        'student': ['学生', '本科生', '研究生', '博士生', '同学', '新生', '毕业生',
                    '学习', '考试', '作业', '论文', '实习', '就读', '入学'],
        'department': ['学院', '系', '专业', '院系', '工学院', '理学院', '人文学院',
                       '计算机科学', '数学', '物理', '软件工程', '信息'],
        'building': ['楼', '教室', '图书馆', '实验室', '宿舍', '食堂', '体育馆',
                     '主楼', '教学楼', '科技楼', '行政楼', '场所', '校园'],
        'research': ['科研', '研究', '项目', '论文', '实验', '成果', '专利',
                     '学术', '发表', '期刊', '会议', '课题', '基金'],
        'activity': ['活动', '比赛', '竞赛', '典礼', '晚会', '社团', '运动会',
                     '讲座', '交流', '志愿者', '文化节', '招聘']
    }
    
    # 为每个类别生成基础向量
    category_base_vectors = {}
    for category in semantic_categories:
        # 每个类别有一个独特的基础向量
        base = np.random.randn(embedding_dim) * 0.5
        category_base_vectors[category] = base
    
    # 为词汇表中的词分配向量
    for word, idx in vocabulary.word2idx.items():
        if word in (vocabulary.PAD_TOKEN, vocabulary.UNK_TOKEN):
            continue
            
        # 检查词语属于哪个类别
        for category, keywords in semantic_categories.items():
            for keyword in keywords:
                if keyword in word or word in keyword:
                    # 使用类别基础向量 + 小随机扰动
                    noise = np.random.randn(embedding_dim) * 0.1
                    embedding_matrix[idx] = category_base_vectors[category] + noise
                    break
            else:
                continue
            break
    
    # 归一化
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embedding_matrix = embedding_matrix / norms
    
    return embedding_matrix


# 注意: download_pretrained_embeddings 函数已移除，因为我们使用语义类别初始化方法
# 不需要下载外部预训练词向量


if __name__ == "__main__":
    # 测试
    from preprocessor import Vocabulary
    
    # 创建测试词汇表
    vocab = Vocabulary()
    test_words = ['计算机', '数据结构', '教授', '学生', '图书馆', '科研', '活动']
    for word in test_words:
        vocab.add_word(word)
    
    print(f"词汇表大小: {len(vocab)}")
    
    # 生成校园领域词向量
    embedding_matrix = generate_campus_embeddings(vocab, embedding_dim=64)
    print(f"嵌入矩阵形状: {embedding_matrix.shape}")
    
    # 测试相似度
    from numpy.linalg import norm
    
    def cosine_sim(v1, v2):
        return np.dot(v1, v2) / (norm(v1) * norm(v2))
    
    idx1 = vocab.get_index('计算机')
    idx2 = vocab.get_index('数据结构')
    idx3 = vocab.get_index('图书馆')
    
    print(f"\n'计算机' vs '数据结构' 相似度: {cosine_sim(embedding_matrix[idx1], embedding_matrix[idx2]):.4f}")
    print(f"'计算机' vs '图书馆' 相似度: {cosine_sim(embedding_matrix[idx1], embedding_matrix[idx3]):.4f}")
