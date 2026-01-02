# -*- coding: utf-8 -*-
"""
主程序模块 - 命令行交互界面
"""

import os
import sys
import argparse
from typing import Optional

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CampusDataset
from preprocessor import ChineseTokenizer, Vocabulary
from model import EmbeddingModel
from trainer import Trainer, train_word_embedding
from embedding_utils import EmbeddingUtils
from visualizer import EmbeddingVisualizer


def print_banner():
    """打印程序横幅"""
    print("=" * 60)
    print("       中文校园领域词嵌入系统")
    print("       Chinese Campus Word Embedding System")
    print("=" * 60)
    print()


def print_help():
    """打印帮助信息"""
    print("""
可用命令:
  vector <词语>          - 获取词语的嵌入向量
  similar <词语> [数量]  - 查找相似词语（默认10个）
  similarity <词1> <词2> - 计算两个词语的相似度
  analogy <A> <B> <C>    - 词语类比：A之于B，如同C之于？
  visualize [方法]       - 可视化词向量（tsne/pca）
  cluster                - 按类别可视化词向量
  export                 - 导出词向量（用于TensorFlow Projector）
  vocab                  - 显示词汇表信息
  help                   - 显示帮助信息
  quit/exit              - 退出程序
""")


def load_model_and_vocab(model_path: str, vocab_path: str):
    """加载模型和词汇表
    
    Args:
        model_path: 模型路径
        vocab_path: 词汇表路径
        
    Returns:
        (model, vocabulary) 元组
    """
    # 加载词汇表
    vocabulary = Vocabulary()
    vocabulary.load(vocab_path)
    
    # 检查是哪种模型
    config_path = model_path.rstrip('/') + '_config.json'
    import json
    
    model = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if 'window_size' in config:
            # Skip-gram模型
            from skipgram_model import SkipGramModel
            model = SkipGramModel(vocab_size=len(vocabulary))
            model.load(model_path)
        else:
            # 分类模型
            model = EmbeddingModel(vocab_size=len(vocabulary))
            model.load(model_path)
    else:
        # 默认尝试加载分类模型
        model = EmbeddingModel(vocab_size=len(vocabulary))
        model.load(model_path)
    
    return model, vocabulary


def interactive_mode(embedding_utils: EmbeddingUtils, 
                     visualizer: EmbeddingVisualizer,
                     output_dir: str = "output") -> None:
    """交互模式
    
    Args:
        embedding_utils: 词向量工具
        visualizer: 可视化器
        output_dir: 输出目录
    """
    print_help()
    
    while True:
        try:
            user_input = input("\n请输入命令 > ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split()
            command = parts[0].lower()
            args = parts[1:]
            
            if command in ('quit', 'exit', 'q'):
                print("再见！")
                break
            
            elif command == 'help':
                print_help()
            
            elif command == 'vector':
                if not args:
                    print("用法: vector <词语>")
                    continue
                word = args[0]
                if embedding_utils.vocabulary.contains(word):
                    vec = embedding_utils.get_word_vector(word)
                    print(f"词语 '{word}' 的嵌入向量:")
                    print(f"  维度: {vec.shape[0]}")
                    print(f"  向量: {vec[:10]}...")  # 只显示前10个值
                else:
                    print(f"词语 '{word}' 不在词汇表中，将使用 <UNK> 向量")
                    vec = embedding_utils.get_word_vector(word)
                    print(f"  向量: {vec[:10]}...")
            
            elif command == 'similar':
                if not args:
                    print("用法: similar <词语> [数量]")
                    continue
                word = args[0]
                top_k = int(args[1]) if len(args) > 1 else 10
                
                similar_words = embedding_utils.find_similar_words(word, top_k)
                if similar_words:
                    print(f"与 '{word}' 最相似的 {len(similar_words)} 个词语:")
                    for w, sim in similar_words:
                        print(f"  {w}: {sim:.4f}")
                else:
                    print(f"未找到与 '{word}' 相似的词语")
            
            elif command == 'similarity':
                if len(args) < 2:
                    print("用法: similarity <词1> <词2>")
                    continue
                word1, word2 = args[0], args[1]
                sim = embedding_utils.cosine_similarity(word1, word2)
                print(f"'{word1}' 和 '{word2}' 的余弦相似度: {sim:.4f}")
            
            elif command == 'analogy':
                if len(args) < 3:
                    print("用法: analogy <A> <B> <C>")
                    print("  计算: A之于B，如同C之于？")
                    continue
                word_a, word_b, word_c = args[0], args[1], args[2]
                results = embedding_utils.word_analogy(word_a, word_b, word_c)
                if results:
                    print(f"'{word_a}' 之于 '{word_b}'，如同 '{word_c}' 之于:")
                    for w, sim in results:
                        print(f"  {w}: {sim:.4f}")
                else:
                    print("未找到类比结果")
            
            elif command == 'visualize':
                method = args[0] if args else 'tsne'
                save_path = os.path.join(output_dir, f"embeddings_{method}.png")
                print(f"正在生成 {method.upper()} 可视化...")
                visualizer.plot_embeddings(method=method, save_path=save_path)
                print(f"可视化已保存到 {save_path}")
            
            elif command == 'cluster':
                # 预定义的词语类别
                categories = {
                    "课程": ["数据结构", "课程", "学习", "教学", "线性代数", "操作系统"],
                    "人员": ["学生", "教授", "老师", "同学", "研究生", "本科生"],
                    "场所": ["图书馆", "教室", "实验室", "宿舍", "食堂", "体育馆"],
                    "院系": ["计算机", "数学", "物理", "学院", "系"],
                    "科研": ["研究", "项目", "论文", "实验", "科研"],
                }
                save_path = os.path.join(output_dir, "clusters.png")
                print("正在生成聚类可视化...")
                visualizer.plot_word_clusters(categories=categories, save_path=save_path)
                print(f"聚类可视化已保存到 {save_path}")
            
            elif command == 'export':
                export_dir = os.path.join(output_dir, "embeddings")
                embedding_utils.export_embeddings(export_dir)
            
            elif command == 'vocab':
                vocab = embedding_utils.vocabulary
                print(f"词汇表信息:")
                print(f"  总词数: {len(vocab)}")
                words = vocab.get_all_words()
                print(f"  示例词语: {words[:20]}...")
            
            else:
                print(f"未知命令: {command}")
                print("输入 'help' 查看可用命令")
                
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")


def train_mode(args) -> None:
    """训练模式
    
    Args:
        args: 命令行参数
    """
    method = getattr(args, 'method', 'skipgram')
    
    if method == 'skipgram':
        print("开始训练Skip-gram词嵌入模型（无监督学习）...")
        from skipgram_model import train_skipgram
        
        model, vocabulary = train_skipgram(
            data_path=args.data_path,
            model_save_path=args.model_path,
            vocab_save_path=args.vocab_path,
            embedding_dim=args.embedding_dim,
            window_size=getattr(args, 'window_size', 3),
            num_negative=getattr(args, 'num_negative', 5),
            epochs=args.epochs,
            batch_size=args.batch_size if args.batch_size > 32 else 256,  # Skip-gram需要更大的batch
            learning_rate=args.learning_rate,
            use_pretrained=args.use_pretrained
        )
    else:
        print("开始训练分类词嵌入模型（有监督学习）...")
        model, vocabulary = train_word_embedding(
            data_path=args.data_path,
            model_save_path=args.model_path,
            vocab_save_path=args.vocab_path,
            embedding_dim=args.embedding_dim,
            max_length=args.max_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_pretrained=args.use_pretrained
        )
    
    print("\n训练完成！")
    print(f"模型已保存到: {args.model_path}")
    print(f"词汇表已保存到: {args.vocab_path}")


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description='中文校园领域词嵌入系统',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    
    # 训练模式
    train_parser = subparsers.add_parser('train', help='训练词嵌入模型')
    train_parser.add_argument('--data-path', type=str, default='data/campus_dataset.json',
                              help='数据集路径')
    train_parser.add_argument('--model-path', type=str, default='models/embedding_model',
                              help='模型保存路径')
    train_parser.add_argument('--vocab-path', type=str, default='models/vocabulary.json',
                              help='词汇表保存路径')
    train_parser.add_argument('--embedding-dim', type=int, default=64,
                              help='嵌入向量维度')
    train_parser.add_argument('--max-length', type=int, default=50,
                              help='序列最大长度')
    train_parser.add_argument('--epochs', type=int, default=30,
                              help='训练轮数')
    train_parser.add_argument('--batch-size', type=int, default=8,
                              help='批次大小')
    train_parser.add_argument('--learning-rate', type=float, default=0.0005,
                              help='学习率')
    train_parser.add_argument('--use-pretrained', action='store_true', default=True,
                              help='使用预训练词向量')
    train_parser.add_argument('--no-pretrained', action='store_false', dest='use_pretrained',
                              help='不使用预训练词向量')
    train_parser.add_argument('--method', type=str, default='skipgram', choices=['skipgram', 'classification'],
                              help='训练方法: skipgram(无监督) 或 classification(有监督)')
    train_parser.add_argument('--window-size', type=int, default=3,
                              help='Skip-gram窗口大小')
    train_parser.add_argument('--num-negative', type=int, default=5,
                              help='Skip-gram负采样数量')
    
    # 交互模式
    interact_parser = subparsers.add_parser('interact', help='交互式探索词嵌入')
    interact_parser.add_argument('--model-path', type=str, default='models/embedding_model',
                                 help='模型路径')
    interact_parser.add_argument('--vocab-path', type=str, default='models/vocabulary.json',
                                 help='词汇表路径')
    interact_parser.add_argument('--output-dir', type=str, default='output',
                                 help='输出目录')
    
    # 可视化模式
    viz_parser = subparsers.add_parser('visualize', help='可视化词嵌入')
    viz_parser.add_argument('--model-path', type=str, default='models/embedding_model',
                            help='模型路径')
    viz_parser.add_argument('--vocab-path', type=str, default='models/vocabulary.json',
                            help='词汇表路径')
    viz_parser.add_argument('--output-dir', type=str, default='output',
                            help='输出目录')
    viz_parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'pca'],
                            help='降维方法')
    
    # 导出模式
    export_parser = subparsers.add_parser('export', help='导出词嵌入（用于TensorFlow Projector）')
    export_parser.add_argument('--model-path', type=str, default='models/embedding_model',
                               help='模型路径')
    export_parser.add_argument('--vocab-path', type=str, default='models/vocabulary.json',
                               help='词汇表路径')
    export_parser.add_argument('--output-dir', type=str, default='output/embeddings',
                               help='输出目录')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.mode == 'train':
        train_mode(args)
        
    elif args.mode == 'interact':
        # 检查模型是否存在
        model_file = args.model_path + '.keras' if not args.model_path.endswith('.keras') else args.model_path
        if not os.path.exists(model_file) or not os.path.exists(args.vocab_path):
            print("模型或词汇表不存在，请先训练模型:")
            print("  python main.py train")
            return
        
        # 加载模型
        print("正在加载模型...")
        model, vocabulary = load_model_and_vocab(args.model_path, args.vocab_path)
        
        # 创建工具类
        embedding_utils = EmbeddingUtils(model, vocabulary)
        visualizer = EmbeddingVisualizer(embedding_utils)
        
        # 进入交互模式
        interactive_mode(embedding_utils, visualizer, args.output_dir)
        
    elif args.mode == 'visualize':
        # 检查模型是否存在
        model_file = args.model_path + '.keras' if not args.model_path.endswith('.keras') else args.model_path
        if not os.path.exists(model_file) or not os.path.exists(args.vocab_path):
            print("模型或词汇表不存在，请先训练模型")
            return
        
        # 加载模型
        print("正在加载模型...")
        model, vocabulary = load_model_and_vocab(args.model_path, args.vocab_path)
        
        # 创建工具类
        embedding_utils = EmbeddingUtils(model, vocabulary)
        visualizer = EmbeddingVisualizer(embedding_utils)
        
        # 生成可视化
        os.makedirs(args.output_dir, exist_ok=True)
        
        print(f"正在生成 {args.method.upper()} 可视化...")
        save_path = os.path.join(args.output_dir, f"embeddings_{args.method}.png")
        visualizer.plot_embeddings(method=args.method, save_path=save_path)
        
        print(f"可视化已保存到 {save_path}")
        
    elif args.mode == 'export':
        # 检查模型是否存在
        model_file = args.model_path + '.keras' if not args.model_path.endswith('.keras') else args.model_path
        if not os.path.exists(model_file) or not os.path.exists(args.vocab_path):
            print("模型或词汇表不存在，请先训练模型")
            return
        
        # 加载模型
        print("正在加载模型...")
        model, vocabulary = load_model_and_vocab(args.model_path, args.vocab_path)
        
        # 创建工具类
        embedding_utils = EmbeddingUtils(model, vocabulary)
        
        # 导出
        embedding_utils.export_embeddings(args.output_dir)
        
    else:
        # 默认：如果没有模型则训练，否则进入交互模式
        model_path = 'models/embedding_model'
        vocab_path = 'models/vocabulary.json'
        model_file = model_path + '.keras'
        
        if not os.path.exists(model_file) or not os.path.exists(vocab_path):
            print("未找到已训练的模型，开始训练...")
            print()
            
            # 创建默认参数
            class DefaultArgs:
                data_path = 'data/campus_dataset.json'
                model_path = model_path
                vocab_path = vocab_path
                embedding_dim = 64
                max_length = 50
                epochs = 30
                batch_size = 8
                learning_rate = 0.0005
                use_pretrained = True
            
            train_mode(DefaultArgs())
        
        # 加载模型并进入交互模式
        print("\n正在加载模型...")
        model, vocabulary = load_model_and_vocab(model_path, vocab_path)
        
        embedding_utils = EmbeddingUtils(model, vocabulary)
        visualizer = EmbeddingVisualizer(embedding_utils)
        
        interactive_mode(embedding_utils, visualizer)


if __name__ == "__main__":
    main()
