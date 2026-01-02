# 中文校园领域词嵌入系统

基于TensorFlow实现的中文校园领域词嵌入（Word Embedding）系统，延续前三个作业的主题（大学词汇XML工具、RDF词汇与SPARQL工具、本体建模与推理）。

## 项目结构

```
4_Chinese_Word_Embedding_Design/
├── dataset.py          # 数据集模块 - 构建校园领域数据集
├── preprocessor.py     # 预处理模块 - 中文分词和词汇表管理
├── model.py            # 模型模块 - TensorFlow词嵌入模型
├── trainer.py          # 训练模块 - 模型训练流程
├── embedding_utils.py  # 工具模块 - 词向量操作和相似度计算
├── visualizer.py       # 可视化模块 - 词向量可视化
├── main.py             # 主程序 - 命令行界面
├── data/               # 数据目录
│   └── campus_dataset.json
├── models/             # 模型目录
│   ├── embedding_model/
│   └── vocabulary.json
└── output/             # 输出目录
    ├── embeddings/     # TSV格式导出
    └── *.png           # 可视化图片
```

## 安装依赖

```bash
# 使用uv安装依赖
uv sync

# 或使用pip
pip install tensorflow jieba numpy matplotlib scikit-learn
```

## 使用方法

### 1. 训练模型

```bash
cd 4_Chinese_Word_Embedding_Design
python main.py train
```

可选参数：
- `--epochs`: 训练轮数（默认20）
- `--embedding-dim`: 嵌入维度（默认64）
- `--batch-size`: 批次大小（默认16）

### 2. 交互模式

```bash
python main.py interact
```

可用命令：
- `vector <词语>` - 获取词语的嵌入向量
- `similar <词语> [数量]` - 查找相似词语
- `similarity <词1> <词2>` - 计算两个词语的相似度
- `analogy <A> <B> <C>` - 词语类比
- `visualize [tsne/pca]` - 可视化词向量
- `cluster` - 按类别可视化词向量
- `export` - 导出词向量
- `help` - 显示帮助

### 3. 可视化

```bash
python main.py visualize --method tsne
```

### 4. 导出词向量

```bash
python main.py export
```

导出的文件可上传到 https://projector.tensorflow.org/ 进行交互式3D可视化。

## 功能特点

1. **数据集构建**：从前三个项目（XML、RDF、本体）提取校园领域数据，生成100+条中文句子
2. **中文分词**：使用jieba进行中文分词，支持标点符号过滤
3. **词嵌入模型**：基于TensorFlow实现，使用文本分类任务学习词嵌入
4. **相似度计算**：支持余弦相似度计算和相似词查找
5. **可视化**：支持t-SNE/PCA降维可视化，支持TensorFlow Projector导出

## 模型架构

```
Embedding Layer (vocab_size × embedding_dim)
    ↓
GlobalAveragePooling1D
    ↓
Dense (64, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (num_classes, Softmax)
```

## 参考资料

- [TensorFlow Word Embeddings Guide](https://www.tensorflow.org/text/guide/word_embeddings)
- [TensorFlow Embedding Projector](https://projector.tensorflow.org/)
