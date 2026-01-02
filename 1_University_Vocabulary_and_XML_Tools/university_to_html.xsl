<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

    <xsl:output method="html" encoding="UTF-8" indent="yes"/>

    <!-- 方便通过 id 查找学院、老师 -->
    <xsl:key name="deptById" match="department" use="@id"/>
    <xsl:key name="teacherById" match="teacher" use="@id"/>

    <xsl:template match="/">
        <html>
            <head>
                <meta charset="UTF-8"/>
                <title>大学信息概览</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
                    th { background-color: #f0f0f0; }
                    h1, h2 { color: #333; }
                </style>
            </head>
            <body>
                <h1>大学信息概览</h1>

                <h2>学院（系）信息</h2>
                <table>
                    <tr>
                        <th>学院编号</th>
                        <th>学院代码</th>
                        <th>学院名称</th>
                        <th>所属大类</th>
                    </tr>
                    <xsl:for-each select="/university/departments/department">
                        <tr>
                            <td><xsl:value-of select="@id"/></td>
                            <td><xsl:value-of select="@code"/></td>
                            <td><xsl:value-of select="name"/></td>
                            <td><xsl:value-of select="faculty"/></td>
                        </tr>
                    </xsl:for-each>
                </table>

                <h2>课程信息</h2>
                <table>
                    <tr>
                        <th>课程编号</th>
                        <th>课程名称</th>
                        <th>学分</th>
                        <th>开课学院</th>
                        <th>授课教师</th>
                    </tr>
                    <xsl:for-each select="/university/courses/course">
                        <tr>
                            <td><xsl:value-of select="@id"/></td>
                            <td><xsl:value-of select="title"/></td>
                            <td><xsl:value-of select="credits"/></td>
                            <td>
                                <xsl:value-of select="key('deptById', departmentRef)/name"/>
                            </td>
                            <td>
                                <xsl:value-of select="key('teacherById', teacherRef)/name"/>
                            </td>
                        </tr>
                    </xsl:for-each>
                </table>

                <h2>学生信息</h2>
                <table>
                    <tr>
                        <th>学号</th>
                        <th>姓名</th>
                        <th>性别</th>
                        <th>入学年份</th>
                        <th>专业</th>
                        <th>所属学院</th>
                    </tr>
                    <xsl:for-each select="/university/students/student">
                        <tr>
                            <td><xsl:value-of select="@id"/></td>
                            <td><xsl:value-of select="name"/></td>
                            <td><xsl:value-of select="gender"/></td>
                            <td><xsl:value-of select="enrollmentYear"/></td>
                            <td><xsl:value-of select="program"/></td>
                            <td>
                                <xsl:value-of
                                    select="key('deptById', @deptRef)/name"/>
                            </td>
                        </tr>
                    </xsl:for-each>
                </table>

            </body>
        </html>
    </xsl:template>
</xsl:stylesheet>
