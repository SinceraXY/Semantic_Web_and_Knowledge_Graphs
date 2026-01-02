import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from rdflib import Graph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VOCAB = os.path.join(BASE_DIR, "cs_vocabulary.ttl")
DEFAULT_DATA = os.path.join(BASE_DIR, "cs_data.ttl")


class SparqlApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RDF / SPARQL 查询演示工具")
        self.root.geometry("960x600")
        self.root.configure(padx=10, pady=10)

        self.graph = Graph()
        self.loaded_files = []

        self._create_widgets()
        self._load_default_graph()

    def _create_widgets(self) -> None:
        # 行 0：已加载 RDF 文件
        tk.Label(self.root, text="已加载的 RDF 文件：").grid(
            row=0, column=0, padx=5, pady=5, sticky="nw"
        )

        self.files_text = scrolledtext.ScrolledText(self.root, width=60, height=4)
        self.files_text.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="we")
        self.files_text.configure(state="disabled")

        tk.Button(
            self.root, text="加载 RDF/TTL 文件", command=self.load_rdf_file
        ).grid(row=0, column=3, padx=5, pady=5, sticky="ne")

        # 行 1：SPARQL 输入
        tk.Label(self.root, text="SPARQL 查询语句：").grid(
            row=1, column=0, padx=5, pady=5, sticky="nw"
        )

        self.query_text = scrolledtext.ScrolledText(self.root, width=80, height=12)
        self.query_text.grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="nsew")

        # 行 2：执行按钮
        tk.Button(
            self.root, text="执行查询", command=self.run_query
        ).grid(row=2, column=3, padx=5, pady=5, sticky="e")

        # 行 3：结果显示
        tk.Label(self.root, text="查询结果：").grid(
            row=3, column=0, padx=5, pady=5, sticky="nw"
        )

        self.result_text = scrolledtext.ScrolledText(self.root, width=80, height=16)
        self.result_text.grid(row=3, column=1, columnspan=3, padx=5, pady=5, sticky="nsew")

        # 网格权重，使文本框可以拉伸
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

        # 预填一条示例查询
        example_query = """PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX cs:   <http://example.org/cs#>

# 示例：查询所有学生及其选修课程
SELECT ?studentName ?courseName
WHERE {
    ?s a cs:Student ;
       cs:name ?studentName ;
       cs:enrolledIn ?course .
    ?course cs:name ?courseName .
}
ORDER BY ?studentName
"""
        self.query_text.insert("1.0", example_query)

    def _append_loaded_file(self, path: str) -> None:
        self.loaded_files.append(path)
        self.files_text.configure(state="normal")
        self.files_text.insert(tk.END, path + "\n")
        self.files_text.configure(state="disabled")

    def _load_default_graph(self) -> None:
        """启动时自动加载内置的词汇和数据。"""
        self.graph = Graph()
        self.loaded_files = []
        self.files_text.configure(state="normal")
        self.files_text.delete("1.0", tk.END)
        self.files_text.configure(state="disabled")

        for path in (DEFAULT_VOCAB, DEFAULT_DATA):
            if os.path.exists(path):
                try:
                    self.graph.parse(path, format="turtle")
                    self._append_loaded_file(path)
                except Exception as e:  # rdflib 解析异常
                    messagebox.showerror("解析错误", f"无法解析文件：{path}\n{e}")

    def load_rdf_file(self) -> None:
        """手动加载其他 RDF/TTL 文件，追加到当前图中。"""
        file_path = filedialog.askopenfilename(
            title="选择 RDF/TTL 文件",
            initialdir=BASE_DIR,
            filetypes=[
                ("Turtle", "*.ttl"),
                ("RDF/XML", "*.rdf;*.xml"),
                ("N-Triples", "*.nt"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            # 不指定 format 让 rdflib 自动猜测；如果失败再尝试 turtle
            try:
                self.graph.parse(file_path)
            except Exception:
                self.graph.parse(file_path, format="turtle")
        except Exception as e:
            messagebox.showerror("解析错误", f"无法解析文件：{file_path}\n{e}")
            return

        self._append_loaded_file(file_path)
        messagebox.showinfo("加载成功", f"已将文件添加到 RDF 图中：\n{file_path}")

    def run_query(self) -> None:
        """执行当前 SPARQL 查询，并在结果区域中显示。"""
        query_str = self.query_text.get("1.0", tk.END).strip()
        if not query_str:
            messagebox.showwarning("查询为空", "请输入 SPARQL 查询语句。")
            return

        # 清空旧结果
        self.result_text.delete("1.0", tk.END)

        try:
            results = self.graph.query(query_str)
        except Exception as e:
            messagebox.showerror("查询错误", str(e))
            return

        # 如果是 SELECT 查询，rdflib 会返回带有变量名的结果集
        if results.type == "SELECT":
            # 先把结果全部取出来，便于统计行数
            rows = list(results)
            headers = [str(v) for v in results.vars]
            header_line = "\t".join(headers)
            self.result_text.insert(tk.END, header_line + "\n")
            self.result_text.insert(tk.END, "-" * max(40, len(header_line)) + "\n")

            if not rows:
                self.result_text.insert(tk.END, "共 0 行结果。\n")
                return

            for row in rows:
                # row 本身是一个有序元组 (val1, val2, ...)
                values = [str(val) if val is not None else "" for val in row]
                line = "\t".join(values)
                self.result_text.insert(tk.END, line + "\n")

            self.result_text.insert(tk.END, f"\n共 {len(rows)} 行结果。\n")
        else:
            # 针对 CONSTRUCT / ASK / DESCRIBE 简单输出字符串表示
            for item in results:
                self.result_text.insert(tk.END, str(item) + "\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = SparqlApp(root)
    root.mainloop()
