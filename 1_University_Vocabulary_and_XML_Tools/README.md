# 大学词汇与 XML 工具

基于 XML 技术的大学信息管理系统，实现了完整的 XML 词汇定义、验证和转换功能。

## 功能特性

- **XML 词汇定义**：定义了大学领域的完整词汇，包括院系、教师、学生、课程、教室等实体
- **DTD 验证**：使用 DTD 定义文档结构约束
- **XML Schema 验证**：使用 XSD 定义更严格的数据类型约束
- **XSLT 转换**：将 XML 数据转换为美观的 HTML 页面
- **XPath 查询**：支持自定义 XPath 表达式查询
- **快速检索**：支持中文关键字快速检索（学生、教师、课程、学院、教室）

## 文件结构

```
1_University_Vocabulary_and_XML_Tools/
├── university.xml              # 大学数据 XML 文件
├── university.dtd              # DTD 文档类型定义
├── university.xsd              # XML Schema 定义
├── university_to_html.xsl      # XSLT 样式表
├── university_xml_app.py       # GUI 应用程序
├── university_output.html      # 转换输出的 HTML
└── university_sample_dtd.xml   # DTD 验证示例
```

## 数据模型

```
University
├── Departments (院系)
│   └── Department: id, code, name, faculty
├── Teachers (教师)
│   └── Teacher: id, deptRef, name, title
├── Students (学生)
│   └── Student: id, deptRef, name, gender, enrollmentYear, program
├── Courses (课程)
│   └── Course: id, title, credits, departmentRef, teacherRef
└── Rooms (教室)
    └── Room: id, name, building, capacity
```

## 使用方法

### 启动 GUI 应用

```bash
cd 1_University_Vocabulary_and_XML_Tools
python university_xml_app.py
```

### 功能说明

1. **加载 XML 文件**：点击"加载 XML 文件"按钮选择要处理的 XML 文件
2. **DTD 验证**：验证 XML 是否符合 DTD 定义的结构
3. **XSD 验证**：验证 XML 是否符合 XML Schema 定义的约束
4. **XSLT 转换**：将 XML 转换为 HTML 并在浏览器中打开
5. **XPath 查询**：输入 XPath 表达式进行自定义查询
6. **快速检索**：输入中文关键字（如"学生"、"课程"）快速检索数据

### XPath 查询示例

```xpath
# 查询所有学生
/university/students/student

# 查询计算机科学系的教师
/university/teachers/teacher[@deptRef='D1']

# 查询学分大于 3 的课程
/university/courses/course[credits > 3]
```

## 技术栈

- Python 3.11+
- lxml（XML 处理库）
- tkinter（GUI 框架）

## 截图预览

应用程序提供了直观的图形界面，支持：
- XML 文件加载与验证
- 结构化数据展示
- 查询结果格式化输出
