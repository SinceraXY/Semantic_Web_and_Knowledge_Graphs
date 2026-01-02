# 本体建模与推理

基于 OWL 本体和 Jena 推理引擎的校园知识图谱系统，支持本体建模、规则推理和 SPARQL 查询。

## 功能特性

- **OWL 本体建模**：使用 OWL 定义校园领域本体
- **规则推理**：基于 Jena 规则引擎进行知识推理
- **SPARQL 查询**：支持对推理前后数据的查询对比
- **Web 界面**：提供交互式 Web 查询界面

## 项目结构

```
3_Ontology_Modeling_and_Reasoning/
├── campus_ontology.ttl      # OWL 本体定义
├── campus_instances.ttl     # 实例数据
├── campus.rules             # Jena 推理规则
├── pom.xml                  # Maven 配置
└── src/
    └── main/
        ├── java/org/example/
        │   ├── App.java         # 命令行应用
        │   └── WebApp.java      # Web 应用
        └── resources/web/
            ├── index.html       # Web 界面
            ├── app.js           # 前端逻辑
            └── styles.css       # 样式表
```

## 本体模型

### 类层次结构

```
AcademicEntity (学术实体)
├── Person (人员)
│   ├── Teacher (教师)
│   └── Student (学生)
│       ├── UndergraduateStudent (本科生)
│       ├── GraduateStudent (研究生)
│       └── ExcellentStudent (优秀学生) [推理得到]
├── Course (课程)
│   ├── UndergraduateCourse (本科课程)
│   └── GraduateCourse (研究生课程)
├── University (大学)
├── Department (学院/系)
├── Building (教学楼)
├── Room (教室)
├── Semester (学期)
├── CourseOffering (课程开课安排)
├── Enrollment (选课记录)
└── Exam (考试)
```

### 推理规则示例

```
# 规则：学生选修某课程，该课程由某教师教授 → 学生被该教师教授
[isTaughtBy:
    (?student cs:enrolledIn ?course)
    (?teacher cs:teaches ?course)
    -> (?student cs:isTaughtBy ?teacher)
]

# 规则：学生成绩优秀 → 标记为优秀学生，具备奖学金资格
[excellentStudent:
    (?enrollment cs:enrollmentGrade ?grade)
    greaterThan(?grade, 90)
    (?enrollment cs:enrollmentOfStudent ?student)
    -> (?student rdf:type cs:ExcellentStudent)
       (?student cs:eligibleForScholarship "true"^^xsd:boolean)
]
```

## 使用方法

### 命令行运行

```bash
cd 3_Ontology_Modeling_and_Reasoning
mvn compile exec:java -Dexec.mainClass="org.example.App"
```

### Web 界面运行

```bash
mvn compile exec:java -Dexec.mainClass="org.example.WebApp"
# 访问 http://localhost:8080
```

### 输出示例

```
=== Base Model loaded ===
Triples: 156

=== Ontology: Classes (in namespace) ===
- AcademicEntity
- Course
- Department
- Student
- Teacher
...

=== SPARQL Query (inferred): who is taught by which teacher ===
| student | teacher |
|---------|---------|
| 张晨    | 王晓    |
| 刘洋    | 赵敏    |
...

=== SPARQL Query (inferred): excellent students and scholarship ===
| student | flag |
|---------|------|
| 张晨    | true |
...
```

## 技术栈

- Java 11+
- Apache Jena（本体处理与推理）
- Maven（项目构建）
- Spark Java（Web 框架）

## 依赖配置

```xml
<dependencies>
    <dependency>
        <groupId>org.apache.jena</groupId>
        <artifactId>apache-jena-libs</artifactId>
        <version>4.10.0</version>
        <type>pom</type>
    </dependency>
</dependencies>
```
