# -*- coding: utf-8 -*-
"""
可视化模块 - 词向量可视化功能
"""

import os
import numpy as np
from typing import List, Dict, Optional

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from embedding_utils import EmbeddingUtils
from preprocessor import Vocabulary


def setup_chinese_font():
    """设置中文字体 - 每次绑图前调用"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'


class EmbeddingVisualizer:
    """词向量可视化器"""
    
    def __init__(self, embedding_utils: EmbeddingUtils):
        """初始化可视化器
        
        Args:
            embedding_utils: 词向量工具类实例
        """
        self.embedding_utils = embedding_utils
        self._reduced_embeddings: Optional[np.ndarray] = None
        self._reduced_words: Optional[List[str]] = None
        
    def reduce_dimensions(self, words: List[str] = None, 
                          method: str = 'tsne',
                          n_components: int = 2,
                          perplexity: int = 30,
                          random_state: int = 42) -> np.ndarray:
        """使用t-SNE或PCA降维"""
        if words is None:
            words = self.embedding_utils.get_vocabulary_words()
        
        valid_words = [w for w in words 
                       if self.embedding_utils.vocabulary.contains(w)]
        
        if len(valid_words) == 0:
            raise ValueError("没有有效的词语可以降维")
        
        embeddings = np.array([
            self.embedding_utils.get_word_vector(w) for w in valid_words
        ])
        
        if method.lower() == 'tsne':
            actual_perplexity = min(perplexity, len(valid_words) - 1)
            actual_perplexity = max(actual_perplexity, 5)
            reducer = TSNE(n_components=n_components, perplexity=actual_perplexity,
                          random_state=random_state, max_iter=1000)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=random_state)
        else:
            raise ValueError(f"不支持的降维方法: {method}")
        
        reduced = reducer.fit_transform(embeddings)
        self._reduced_embeddings = reduced
        self._reduced_words = valid_words
        return reduced
    
    def plot_embeddings(self, words: List[str] = None, 
                        method: str = 'tsne',
                        save_path: str = None,
                        figsize: tuple = (14, 10),
                        fontsize: int = 9,
                        title: str = None,
                        max_words: int = 80) -> None:
        """绘制词向量散点图"""
        # 设置中文字体
        setup_chinese_font()
        
        if words is None:
            all_words = self.embedding_utils.get_vocabulary_words()
            filtered_words = [w for w in all_words 
                            if len(w) > 1 and w not in ['<PAD>', '<UNK>']]
            words = filtered_words[:max_words]
        
        reduced = self.reduce_dimensions(words, method=method)
        valid_words = self._reduced_words
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绑制散点
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], 
                           c=range(len(valid_words)), cmap='viridis',
                           alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
        
        # 添加词语标签
        for i, word in enumerate(valid_words):
            ax.annotate(word, (reduced[i, 0], reduced[i, 1]),
                       fontsize=fontsize, alpha=0.85,
                       xytext=(3, 3), textcoords='offset points')
        
        if title is None:
            title = f'校园领域词向量可视化 ({method.upper()})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('维度1', fontsize=11)
        ax.set_ylabel('维度2', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"可视化图片已保存到 {save_path}")
        
        plt.close()
    
    def plot_word_clusters(self, categories: Dict[str, List[str]], 
                           method: str = 'tsne',
                           save_path: str = None,
                           figsize: tuple = (14, 10),
                           fontsize: int = 10,
                           title: str = None) -> None:
        """按类别绘制词向量聚类图"""
        # 设置中文字体
        setup_chinese_font()
        
        all_words = []
        word_categories = {}
        
        for category, words in categories.items():
            for word in words:
                if self.embedding_utils.vocabulary.contains(word):
                    all_words.append(word)
                    word_categories[word] = category
        
        if len(all_words) == 0:
            raise ValueError("没有有效的词语可以可视化")
        
        reduced = self.reduce_dimensions(all_words, method=method)
        valid_words = self._reduced_words
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 颜色调色板
        unique_categories = list(categories.keys())
        color_palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
                        '#1abc9c', '#e67e22', '#34495e']
        category_colors = {cat: color_palette[i % len(color_palette)] 
                          for i, cat in enumerate(unique_categories)}
        
        for category in unique_categories:
            indices = [i for i, w in enumerate(valid_words) 
                       if word_categories.get(w) == category]
            
            if indices:
                cat_reduced = reduced[indices]
                ax.scatter(cat_reduced[:, 0], cat_reduced[:, 1],
                          c=category_colors[category], label=category,
                          alpha=0.8, s=100, edgecolors='white', linewidth=1)
                
                for idx in indices:
                    ax.annotate(valid_words[idx],
                               (reduced[idx, 0], reduced[idx, 1]),
                               fontsize=fontsize, fontweight='medium',
                               xytext=(5, 5), textcoords='offset points')
        
        ax.legend(loc='upper right', fontsize=10, title='词语类别')
        
        if title is None:
            title = f'校园领域词向量聚类可视化 ({method.upper()})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('维度1', fontsize=11)
        ax.set_ylabel('维度2', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"聚类可视化图片已保存到 {save_path}")
        
        plt.close()
    
    def plot_similarity_heatmap(self, words: List[str],
                                save_path: str = None,
                                figsize: tuple = (10, 8),
                                title: str = None) -> None:
        """绘制词语相似度热力图"""
        # 设置中文字体
        setup_chinese_font()
        
        valid_words = [w for w in words 
                       if self.embedding_utils.vocabulary.contains(w)]
        
        if len(valid_words) < 2:
            raise ValueError("需要至少2个有效词语")
        
        n = len(valid_words)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = self.embedding_utils.cosine_similarity(
                        valid_words[i], valid_words[j])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(valid_words, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(valid_words, fontsize=10)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('余弦相似度', fontsize=11)
        
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                       ha='center', va='center', fontsize=9,
                       color='white' if abs(similarity_matrix[i, j]) > 0.5 else 'black')
        
        if title is None:
            title = '词语相似度热力图'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"相似度热力图已保存到 {save_path}")
        
        plt.close()
