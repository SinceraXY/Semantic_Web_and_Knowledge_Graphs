# 语义WEB与知识图谱课程项目

<p align="center">
  <strong>Semantic Web and Knowledge Graphs Course Projects</strong>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> •
  <a href="#项目概览">项目概览</a> •
  <a href="#技术栈">技术栈</a> •
  <a href="#许可证">许可证</a>
</p>

---

本项目是语义网与知识图谱课程的系列实践项目，围绕校园领域构建了从 XML 到 RDF、本体建模、推理以及词嵌入的完整知识表示与处理流程。

## 项目概览

| 项目 | 描述 | 技术栈 |
|:----:|------|--------|
| [1. 大学词汇与 XML 工具](./1_University_Vocabulary_and_XML_Tools) | XML 词汇定义、DTD/XSD 验证、XSLT 转换 | Python, lxml |
| [2. RDF 词汇与 SPARQL 工具](./2_RDF_Vocabulary_and_SPARQL_Tool) | RDFS 词汇定义、RDF 数据管理、SPARQL 查询 | Python, rdflib |
| [3. 本体建模与推理](./3_Ontology_Modeling_and_Reasoning) | OWL 本体、Jena 规则推理、知识图谱查询 | Java, Apache Jena |
| [4. 中文词嵌入设计](./4_Chinese_Word_Embedding_Design) | 中文校园领域词嵌入、Skip-gram 模型 | Python, TensorFlow |

## 项目架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    语义网与知识图谱技术栈                        │
├─────────────────────────────────────────────────────────────────┤
│  1. 数据层      │  XML        │  DTD/XSD   │  数据验证与转换     │
├─────────────────────────────────────────────────────────────────┤
│  2. 语义层      │  RDF/RDFS   │  SPARQL    │  知识图谱查询       │
├─────────────────────────────────────────────────────────────────┤
│  3. 推理层      │  OWL 本体   │  规则推理   │  知识推断          │
├─────────────────────────────────────────────────────────────────┤
│  4. 词嵌入层    │  中文词嵌入  │  Skip-gram  │  语义相似度计算    │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求

- Python 3.11+
- Java 11+（项目 3）
- Maven（项目 3）

### 安装依赖

```bash
# 使用 uv 安装 Python 依赖
uv sync

# 或使用 pip
pip install lxml rdflib tensorflow jieba numpy matplotlib scikit-learn
```

### 运行各项目

```bash
# 1. XML 工具
cd 1_University_Vocabulary_and_XML_Tools
python university_xml_app.py

# 2. SPARQL 工具
cd 2_RDF_Vocabulary_and_SPARQL_Tool
python sparql_gui.py

# 3. 本体推理（需要 Maven）
cd 3_Ontology_Modeling_and_Reasoning
mvn compile exec:java -Dexec.mainClass="org.example.App"

# 4. 词嵌入系统
cd 4_Chinese_Word_Embedding_Design
python main.py train
python main.py interact
```

## 项目详情

### 1. 大学词汇与 XML 工具

定义了大学领域的 XML 词汇，包括院系、教师、学生、课程、教室等实体。提供 GUI 工具支持：
- DTD/XML Schema 验证
- XSLT 转换为 HTML
- XPath 查询
- 中文关键字快速检索

### 2. RDF 词汇与 SPARQL 工具

使用 RDFS 定义计算机科学与校园领域词汇，支持：
- 完整的类层次结构定义
- 对象属性和数据属性定义
- SPARQL 查询界面
- 多 RDF 文件加载

### 3. 本体建模与推理

基于 OWL 和 Apache Jena 构建校园知识图谱：
- OWL 本体建模
- 自定义推理规则
- 推理前后数据对比
- Web 查询界面

### 4. 中文词嵌入设计

基于 TensorFlow 的中文校园领域词嵌入系统：
- Skip-gram 无监督学习
- 词向量相似度计算
- t-SNE/PCA 可视化
- TensorFlow Projector 导出

## 数据模型

四个项目共享统一的校园领域数据模型：

```
校园领域模型
├── 组织结构
│   ├── 大学 (University)
│   ├── 学院 (Faculty)
│   └── 院系 (Department)
├── 人员
│   ├── 教师 (Teacher)
│   └── 学生 (Student)
├── 教学
│   ├── 课程 (Course)
│   ├── 选课 (Enrollment)
│   └── 考试 (Exam)
└── 设施
    ├── 教学楼 (Building)
    └── 教室 (Room)
```

## 技术栈

| 技术 | 用途 |
|------|------|
| XML/DTD/XSD | 数据结构定义与验证 |
| XSLT | XML 转换 |
| RDF/RDFS | 语义数据表示 |
| SPARQL | RDF 查询语言 |
| OWL | 本体建模 |
| Apache Jena | Java RDF/OWL 处理框架 |
| TensorFlow | 深度学习框架 |
| jieba | 中文分词 |

## 项目结构

```
Semantic_Web_and_Knowledge_Graphs/
├── 1_University_Vocabulary_and_XML_Tools/   # XML 词汇与工具
│   ├── university.xml                       # 大学数据
│   ├── university.dtd                       # DTD 定义
│   ├── university.xsd                       # XML Schema
│   ├── university_to_html.xsl               # XSLT 样式表
│   └── university_xml_app.py                # GUI 应用
├── 2_RDF_Vocabulary_and_SPARQL_Tool/        # RDF 词汇与 SPARQL
│   ├── cs_vocabulary.ttl                    # RDFS 词汇
│   ├── cs_data.ttl                          # RDF 数据
│   └── sparql_gui.py                        # SPARQL 查询工具
├── 3_Ontology_Modeling_and_Reasoning/       # 本体建模与推理
│   ├── campus_ontology.ttl                  # OWL 本体
│   ├── campus_instances.ttl                 # 实例数据
│   ├── campus.rules                         # 推理规则
│   └── src/                                 # Java 源码
├── 4_Chinese_Word_Embedding_Design/         # 中文词嵌入
│   ├── main.py                              # 主程序
│   ├── model.py                             # 模型定义
│   ├── skipgram_model.py                    # Skip-gram 模型
│   └── data/                                # 数据集
├── pyproject.toml                           # Python 项目配置
├── LICENSE                                  # 许可证
└── README.md                                # 项目说明
```

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- [Apache Jena](https://jena.apache.org/) - RDF/OWL 处理框架
- [rdflib](https://rdflib.readthedocs.io/) - Python RDF 库
- [TensorFlow](https://www.tensorflow.org/) - 深度学习框架
- [jieba](https://github.com/fxsjy/jieba) - 中文分词库
