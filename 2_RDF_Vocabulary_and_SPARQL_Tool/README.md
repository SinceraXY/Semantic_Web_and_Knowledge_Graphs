# RDF 词汇与 SPARQL 查询工具

基于 RDF/RDFS 的计算机科学领域知识表示系统，提供完整的词汇定义和 SPARQL 查询功能。

## 功能特性

- **RDFS 词汇定义**：定义了计算机科学与校园领域的完整词汇体系
- **RDF 数据管理**：使用 Turtle 格式存储结构化数据
- **SPARQL 查询**：支持标准 SPARQL 查询语言
- **图形界面**：提供友好的 GUI 查询工具

## 文件结构

```
2_RDF_Vocabulary_and_SPARQL_Tool/
├── cs_vocabulary.ttl    # RDFS 词汇定义
├── cs_data.ttl          # RDF 实例数据
└── sparql_gui.py        # SPARQL 查询 GUI 工具
```

## 词汇体系

### 核心类层次

```
AcademicEntity (学术实体)
├── Person (人员)
│   ├── Teacher (教师)
│   ├── Student (学生)
│   │   ├── UndergraduateStudent (本科生)
│   │   └── GraduateStudent (研究生)
│   └── Staff (职员)
│       ├── Administrator (行政人员)
│       └── Advisor (辅导员/导师)
├── Course (课程)
│   ├── UndergraduateCourse (本科课程)
│   └── GraduateCourse (研究生课程)
├── University (大学)
├── Department (学院/系)
├── Building (教学楼)
├── Room (教室)
├── ResearchProject (科研项目)
└── Publication (出版物)
```

### 主要属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `cs:teaches` | 对象属性 | 教师 → 课程 |
| `cs:enrolledIn` | 对象属性 | 学生 → 课程 |
| `cs:supervises` | 对象属性 | 教师 → 科研项目 |
| `cs:belongsToDepartment` | 对象属性 | 实体 → 院系 |
| `cs:name` | 数据属性 | 名称 |
| `cs:credits` | 数据属性 | 学分 |
| `cs:enrollmentYear` | 数据属性 | 入学年份 |

## 使用方法

### 启动 SPARQL 查询工具

```bash
cd 2_RDF_Vocabulary_and_SPARQL_Tool
python sparql_gui.py
```

### SPARQL 查询示例

```sparql
# 查询所有学生及其选修课程
PREFIX cs: <http://example.org/cs#>
SELECT ?studentName ?courseName
WHERE {
    ?s a cs:Student ;
       cs:name ?studentName ;
       cs:enrolledIn ?course .
    ?course cs:name ?courseName .
}
ORDER BY ?studentName

# 查询教师及其教授的课程
PREFIX cs: <http://example.org/cs#>
SELECT ?teacherName ?courseName
WHERE {
    ?t a cs:Teacher ;
       cs:name ?teacherName ;
       cs:teaches ?course .
    ?course cs:name ?courseName .
}

# 查询某院系的所有成员
PREFIX cs: <http://example.org/cs#>
SELECT ?personName ?deptName
WHERE {
    ?p cs:belongsToDepartment ?d ;
       cs:name ?personName .
    ?d cs:name ?deptName .
}
```

## 技术栈

- Python 3.11+
- rdflib（RDF 处理库）
- tkinter（GUI 框架）

## 命名空间

```turtle
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix cs:   <http://example.org/cs#> .
```
