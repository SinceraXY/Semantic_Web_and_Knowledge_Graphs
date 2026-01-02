import os
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from lxml import etree

# 当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DTD_FILE = os.path.join(BASE_DIR, "university.dtd")
XSD_FILE = os.path.join(BASE_DIR, "university.xsd")
XSLT_FILE = os.path.join(BASE_DIR, "university_to_html.xsl")
DEFAULT_XML_FILE = os.path.join(BASE_DIR, "university.xml")


class UniversityXmlApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("大学 XML 词汇演示系统")
        self.root.geometry("960x600")
        self.root.configure(padx=10, pady=10)

        self.xml_tree = None
        self.xml_file_path = None
        self.schema = None  # 缓存 XSD schema
        self.dtd = None     # 缓存 DTD

        self.xml_path_var = tk.StringVar()

        self._create_widgets()
        self._try_load_default_xml()

    def _create_widgets(self) -> None:
        # 行 0：XML 文件路径 + 按钮
        tk.Label(self.root, text="当前 XML 文件：").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )

        entry = tk.Entry(
            self.root, textvariable=self.xml_path_var, width=60, state="readonly"
        )
        entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")

        tk.Button(
            self.root, text="加载 XML 文件", command=self.load_xml
        ).grid(row=0, column=2, padx=5, pady=5)

        # 行 1：验证、转换按钮
        tk.Button(
            self.root, text="使用 DTD 验证 XML", command=self.validate_xml_dtd
        ).grid(row=1, column=0, padx=5, pady=5, sticky="w")

        tk.Button(
            self.root, text="使用 XML Schema 验证 XML", command=self.validate_xml_xsd
        ).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        tk.Button(
            self.root, text="XML 转 HTML（XSLT）", command=self.transform_to_html
        ).grid(row=1, column=2, padx=5, pady=5, sticky="w")

        # 行 2：XPath 输入
        tk.Label(self.root, text="XPath 表达式：").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        self.xpath_entry = tk.Entry(self.root, width=60)
        self.xpath_entry.grid(row=2, column=1, padx=5, pady=5, sticky="we")

        tk.Button(
            self.root, text="执行 XPath 查询", command=self.run_xpath_query
        ).grid(row=2, column=2, padx=5, pady=5)

        # 行 3：快速检索
        tk.Label(self.root, text="快速检索关键字：").grid(
            row=3, column=0, padx=5, pady=5, sticky="w"
        )
        self.search_entry = tk.Entry(self.root, width=60)
        self.search_entry.grid(row=3, column=1, padx=5, pady=5, sticky="we")

        tk.Button(
            self.root, text="执行快速检索", command=self.quick_search
        ).grid(row=3, column=2, padx=5, pady=5)

        # 行 4：查询 / 检索结果
        tk.Label(self.root, text="查询 / 检索结果：").grid(
            row=4, column=0, padx=5, pady=5, sticky="nw"
        )

        self.result_text = scrolledtext.ScrolledText(self.root, width=80, height=20)
        self.result_text.grid(
            row=4, column=1, columnspan=2, padx=5, pady=5, sticky="nsew"
        )

        # 让文本区可拉伸
        self.root.grid_rowconfigure(4, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def _try_load_default_xml(self) -> None:
        """程序启动时尝试自动加载默认 XML。"""
        if os.path.exists(DEFAULT_XML_FILE):
            try:
                self.xml_tree = etree.parse(DEFAULT_XML_FILE)
                self.xml_file_path = DEFAULT_XML_FILE
                self.xml_path_var.set(DEFAULT_XML_FILE)
            except (OSError, etree.XMLSyntaxError):
                # 如果默认文件有问题，静默忽略，让用户手动加载
                self.xml_tree = None
                self.xml_file_path = None

    def load_xml(self) -> None:
        """选择并加载 XML 文件（只检查是否为合法 XML）。"""
        file_path = filedialog.askopenfilename(
            title="选择 XML 文件",
            filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
            initialdir=BASE_DIR,
        )
        if not file_path:
            return

        try:
            tree = etree.parse(file_path)
        except etree.XMLSyntaxError as e:
            messagebox.showerror("XML 解析错误", f"该文件不是合法的 XML：\n{e}")
            self.xml_tree = None
            self.xml_file_path = None
            self.xml_path_var.set("")
            return

        self.xml_tree = tree
        self.xml_file_path = file_path
        self.xml_path_var.set(file_path)
        messagebox.showinfo("加载成功", "XML 文件加载成功，可以进行验证和查询。")

    def _load_schema(self):
        """加载 XSD，返回 XMLSchema 对象。"""
        if self.schema is not None:
            return self.schema

        if not os.path.exists(XSD_FILE):
            messagebox.showerror("缺少 XSD", f"未找到 XSD 文件：\n{XSD_FILE}")
            return None

        try:
            with open(XSD_FILE, "rb") as f:
                schema_doc = etree.parse(f)
            self.schema = etree.XMLSchema(schema_doc)
            return self.schema
        except (OSError, etree.XMLSchemaParseError) as e:
            messagebox.showerror("XSD 加载失败", str(e))
            return None

    def _load_dtd(self):
        """加载 DTD，返回 DTD 对象。"""
        if self.dtd is not None:
            return self.dtd

        if not os.path.exists(DTD_FILE):
            messagebox.showerror("缺少 DTD", f"未找到 DTD 文件：\n{DTD_FILE}")
            return None

        try:
            with open(DTD_FILE, "rb") as f:
                self.dtd = etree.DTD(f)
            return self.dtd
        except (OSError, etree.DTDParseError) as e:
            messagebox.showerror("DTD 加载失败", str(e))
            return None

    def validate_xml_xsd(self) -> None:
        """用 XSD 验证当前 XML，格式不满足时弹出提示。"""
        if self.xml_tree is None:
            messagebox.showwarning("未加载 XML", "请先加载一个 XML 文件。")
            return

        schema = self._load_schema()
        if schema is None:
            return

        try:
            schema.assertValid(self.xml_tree)
        except etree.DocumentInvalid as e:
            # 给出详细错误日志
            log = e.error_log
            msg_lines = ["XML 不符合 XSD：", str(e)]
            if log:
                msg_lines.append("")
                msg_lines.append("错误详情：")
                for entry in log:
                    msg_lines.append(
                        f"- 行 {entry.line}: {entry.message}"
                    )
            messagebox.showerror("XSD 验证失败", "\n".join(msg_lines))
            return

        messagebox.showinfo("XSD 验证通过", "XML 文档符合 university.xsd 定义。")

    def validate_xml_dtd(self) -> None:
        """用 DTD 验证当前 XML，格式不满足时弹出提示。"""
        if self.xml_tree is None:
            messagebox.showwarning("未加载 XML", "请先加载一个 XML 文件。")
            return

        dtd = self._load_dtd()
        if dtd is None:
            return

        is_valid = dtd.validate(self.xml_tree)
        if not is_valid:
            log = dtd.error_log
            msg_lines = ["XML 不符合 DTD："]
            if log:
                msg_lines.append("")
                msg_lines.append("错误详情：")
                for entry in log:
                    msg_lines.append(
                        f"- 行 {entry.line}: {entry.message}"
                    )
            else:
                msg_lines.append("(DTD 验证失败，但未提供详细错误日志。)")

            messagebox.showerror("DTD 验证失败", "\n".join(msg_lines))
            return

        messagebox.showinfo("DTD 验证通过", "XML 文档符合 university.dtd 定义。")

    def quick_search(self) -> None:
        """根据中文关键字进行快速检索，例如：学生、教师、课程、学院、教室。"""
        if self.xml_tree is None:
            messagebox.showwarning("未加载 XML", "请先加载一个 XML 文档。")
            return

        keyword = self.search_entry.get().strip()
        if not keyword:
            messagebox.showwarning(
                "关键字为空",
                "请输入检索关键字，例如：学生、教师、课程、学院、教室。",
            )
            return

        kind = None
        xpath_expr = None

        if "学生" in keyword:
            kind = "student"
            xpath_expr = "/university/students/student"
        elif "教师" in keyword or "老师" in keyword:
            kind = "teacher"
            xpath_expr = "/university/teachers/teacher"
        elif "课程" in keyword:
            kind = "course"
            xpath_expr = "/university/courses/course"
        elif "学院" in keyword or "系" in keyword:
            kind = "department"
            xpath_expr = "/university/departments/department"
        elif "教室" in keyword:
            kind = "room"
            xpath_expr = "/university/rooms/room"
        else:
            messagebox.showinfo(
                "不支持的关键字",
                "目前支持的检索关键字包括：学生、教师、课程、学院、教室。",
            )
            return

        try:
            nodes = self.xml_tree.xpath(xpath_expr)
        except etree.XPathEvalError as e:
            messagebox.showerror("XPath 错误", str(e))
            return

        self.result_text.delete("1.0", tk.END)

        if not nodes:
            self.result_text.insert(tk.END, "未找到任何匹配的数据。\n")
            return

        if kind == "student":
            self._display_students(nodes)
        elif kind == "teacher":
            self._display_teachers(nodes)
        elif kind == "course":
            self._display_courses(nodes)
        elif kind == "department":
            self._display_departments(nodes)
        elif kind == "room":
            self._display_rooms(nodes)

    def run_xpath_query(self) -> None:
        """执行 XPath 查询，并把结果列在文本框中。"""
        if self.xml_tree is None:
            messagebox.showwarning("未加载 XML", "请先加载并验证一个 XML 文档。")
            return

        expr = self.xpath_entry.get().strip()
        if not expr:
            messagebox.showwarning("XPath 为空", "请输入 XPath 表达式。")
            return

        try:
            result = self.xml_tree.xpath(expr)
        except etree.XPathEvalError as e:
            messagebox.showerror("XPath 错误", str(e))
            return

        self.result_text.delete("1.0", tk.END)

        if not result:
            self.result_text.insert(tk.END, "无匹配结果。\n")
            return

        for i, item in enumerate(result, start=1):
            self.result_text.insert(tk.END, f"[{i}]\n")
            if isinstance(item, etree._Element):
                text = etree.tostring(
                    item, pretty_print=True, encoding="unicode"
                )
                self.result_text.insert(tk.END, text + "\n")
            else:
                self.result_text.insert(tk.END, str(item) + "\n\n")

    def _get_department_name(self, dept_id: str) -> str:
        """根据学院编号查找学院名称。"""
        if not dept_id or self.xml_tree is None:
            return ""
        result = self.xml_tree.xpath(
            f"/university/departments/department[@id='{dept_id}']/name/text()"
        )
        return result[0] if result else ""

    def _get_teacher_name(self, teacher_id: str) -> str:
        """根据教师编号查找教师姓名。"""
        if not teacher_id or self.xml_tree is None:
            return ""
        result = self.xml_tree.xpath(
            f"/university/teachers/teacher[@id='{teacher_id}']/name/text()"
        )
        return result[0] if result else ""

    def _display_students(self, nodes) -> None:
        """以结构化的中文格式显示学生列表。"""
        self.result_text.insert(tk.END, "【学生列表】\n")
        for i, stu in enumerate(nodes, start=1):
            sid = stu.get("id", "")
            name = (stu.findtext("name", "") or "").strip()
            gender = (stu.findtext("gender", "") or "").strip()
            year = (stu.findtext("enrollmentYear", "") or "").strip()
            program = (stu.findtext("program", "") or "").strip()
            dept_ref = stu.get("deptRef", "")
            dept_name = self._get_department_name(dept_ref)
            lines = [
                f"{i}. 学号：{sid}",
                f"   姓名：{name}",
                f"   性别：{gender}",
                f"   入学年份：{year}",
                f"   专业：{program}",
                f"   所属学院：{dept_name}（{dept_ref}）" if dept_name else f"   所属学院编号：{dept_ref}",
                "",
            ]
            self.result_text.insert(tk.END, "\n".join(lines) + "\n")

    def _display_teachers(self, nodes) -> None:
        """以结构化的中文格式显示教师列表。"""
        self.result_text.insert(tk.END, "【教师列表】\n")
        for i, t in enumerate(nodes, start=1):
            tid = t.get("id", "")
            name = (t.findtext("name", "") or "").strip()
            title = (t.findtext("title", "") or "").strip()
            dept_ref = t.get("deptRef", "")
            dept_name = self._get_department_name(dept_ref)
            lines = [
                f"{i}. 工号：{tid}",
                f"   姓名：{name}",
                f"   职称：{title}",
                f"   所属学院：{dept_name}（{dept_ref}）" if dept_name else f"   所属学院编号：{dept_ref}",
                "",
            ]
            self.result_text.insert(tk.END, "\n".join(lines) + "\n")

    def _display_courses(self, nodes) -> None:
        """以结构化的中文格式显示课程列表。"""
        self.result_text.insert(tk.END, "【课程列表】\n")
        for i, c in enumerate(nodes, start=1):
            cid = c.get("id", "")
            title = (c.findtext("title", "") or "").strip()
            credits = (c.findtext("credits", "") or "").strip()
            dept_ref = (c.findtext("departmentRef", "") or "").strip()
            teacher_ref = (c.findtext("teacherRef", "") or "").strip()
            dept_name = self._get_department_name(dept_ref)
            teacher_name = self._get_teacher_name(teacher_ref)
            lines = [
                f"{i}. 课程编号：{cid}",
                f"   课程名称：{title}",
                f"   学分：{credits}",
                f"   开课学院：{dept_name}（{dept_ref}）" if dept_name else f"   开课学院编号：{dept_ref}",
                f"   授课教师：{teacher_name}（{teacher_ref}）" if teacher_name else f"   授课教师编号：{teacher_ref}",
                "",
            ]
            self.result_text.insert(tk.END, "\n".join(lines) + "\n")

    def _display_departments(self, nodes) -> None:
        """以结构化的中文格式显示学院/系列表。"""
        self.result_text.insert(tk.END, "【学院 / 系列表】\n")
        for i, d in enumerate(nodes, start=1):
            did = d.get("id", "")
            code = d.get("code", "")
            name = (d.findtext("name", "") or "").strip()
            faculty = (d.findtext("faculty", "") or "").strip()
            lines = [
                f"{i}. 学院编号：{did}",
                f"   学院代码：{code}",
                f"   学院名称：{name}",
                f"   所属大类：{faculty}",
                "",
            ]
            self.result_text.insert(tk.END, "\n".join(lines) + "\n")

    def _display_rooms(self, nodes) -> None:
        """以结构化的中文格式显示教室列表。"""
        self.result_text.insert(tk.END, "【教室列表】\n")
        for i, r in enumerate(nodes, start=1):
            rid = r.get("id", "")
            name = (r.findtext("name", "") or "").strip()
            building = (r.findtext("building", "") or "").strip()
            capacity = (r.findtext("capacity", "") or "").strip()
            lines = [
                f"{i}. 教室编号：{rid}",
                f"   教室名称：{name}",
                f"   所在楼宇：{building}",
                f"   容量：{capacity}",
                "",
            ]
            self.result_text.insert(tk.END, "\n".join(lines) + "\n")

    def transform_to_html(self) -> None:
        """用 XSLT 把当前 XML 转成 HTML，并在浏览器打开。"""
        if self.xml_tree is None:
            messagebox.showwarning("未加载 XML", "请先加载一个 XML 文档。")
            return

        if not os.path.exists(XSLT_FILE):
            messagebox.showerror(
                "缺少 XSLT 文件", f"未找到 XSLT 文件：\n{XSLT_FILE}"
            )
            return

        try:
            xslt_doc = etree.parse(XSLT_FILE)
            transform = etree.XSLT(xslt_doc)
            result_tree = transform(self.xml_tree)
        except (OSError, etree.XSLTParseError, etree.XSLTApplyError) as e:
            messagebox.showerror("XSLT 错误", str(e))
            return

        # HTML 输出路径：与 XML 同目录
        if self.xml_file_path:
            out_dir = os.path.dirname(self.xml_file_path)
        else:
            out_dir = BASE_DIR
        html_path = os.path.join(out_dir, "university_output.html")

        try:
            html_bytes = etree.tostring(
                result_tree,
                pretty_print=True,
                encoding="utf-8",
                method="html",
            )
            with open(html_path, "wb") as f:
                f.write(html_bytes)
        except OSError as e:
            messagebox.showerror("保存 HTML 失败", str(e))
            return

        messagebox.showinfo("转换成功", f"已生成 HTML 文件：\n{html_path}")
        webbrowser.open(f"file://{os.path.abspath(html_path)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = UniversityXmlApp(root)
    root.mainloop()
