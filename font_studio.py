#!/usr/bin/env python3
"""TwinLine Font Studio: build a paired English-over-Chinese TTF without editing JSON."""
import json
import subprocess
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
SAMPLE_EN = "The quick brown fox jumps over the lazy dog.\nTypography is the art of arranging type.\nOne thought, two languages, one visual unit."
SAMPLE_ZH = "敏捷的棕色狐狸跳过了那只懒狗。\n字体排印是安排文字的艺术。\n一种思想，两种语言，一个视觉整体。"

class FontStudio:
    def __init__(self, root):
        self.root = root
        root.title("TwinLine V2 Font Studio")
        root.geometry("980x760")
        root.minsize(680, 560)
        root.configure(bg="#080808")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#080808")
        style.configure("TLabel", background="#080808", foreground="#eeeeee")
        style.configure("TButton", padding=8)
        style.configure("Muted.TLabel", foreground="#aaaaaa")
        outer = ttk.Frame(root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="TwinLine V2 · 双层字体工作台", font=("Arial", 19, "bold")).pack(anchor="w")
        ttk.Label(outer, text="英文一行对应中文一行。超长文本会按字体实际字宽自动拆分，生成多枚有序 TTF 组合字形。", style="Muted.TLabel").pack(anchor="w", pady=(5, 14))

        editors = ttk.Frame(outer)
        editors.pack(fill="both", expand=True)
        editors.columnconfigure(0, weight=1)
        editors.columnconfigure(1, weight=1)
        editors.rowconfigure(1, weight=1)
        ttk.Label(editors, text="英文（每行一条）").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 5))
        ttk.Label(editors, text="中文（顺序必须对应）").grid(row=0, column=1, sticky="w", padx=(8, 0), pady=(0, 5))
        self.en = tk.Text(editors, wrap="word", bg="#151515", fg="#f5f5f5", insertbackground="white", relief="flat", padx=12, pady=12, font=("Arial", 12), undo=True)
        self.zh = tk.Text(editors, wrap="word", bg="#151515", fg="#f5f5f5", insertbackground="white", relief="flat", padx=12, pady=12, font=("Microsoft YaHei", 12), undo=True)
        self.en.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.zh.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        self.en.insert("1.0", SAMPLE_EN)
        self.zh.insert("1.0", SAMPLE_ZH)

        settings = ttk.Frame(outer)
        settings.pack(fill="x", pady=(14, 8))
        ttk.Label(settings, text="字体名称：").pack(side="left")
        self.family = tk.StringVar(value="TwinLine Custom")
        ttk.Entry(settings, textvariable=self.family, width=28).pack(side="left", padx=(4, 18))
        self.count = ttk.Label(settings, text="")
        self.count.pack(side="left")
        ttk.Button(outer, text="检查配对", command=self.check).pack(side="left", pady=(4, 8))
        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(5, 0))
        ttk.Button(actions, text="恢复示例", command=self.reset).pack(side="left")
        ttk.Button(actions, text="生成 TTF 字体", command=self.build).pack(side="right")
        self.status = ttk.Label(outer, text="准备就绪。", style="Muted.TLabel", wraplength=900)
        self.status.pack(anchor="w", pady=(12, 0))
        self.en.bind("<KeyRelease>", lambda _e: self.check(False))
        self.zh.bind("<KeyRelease>", lambda _e: self.check(False))
        self.check(False)

    def get_lines(self, widget):
        value = widget.get("1.0", "end-1c").replace("\r", "")
        return value.split("\n") if value else []

    def check(self, show=True):
        en, zh = self.get_lines(self.en), self.get_lines(self.zh)
        ok = len(en) == len(zh) and bool(en) and all(a.strip() and b.strip() for a, b in zip(en, zh))
        self.count.configure(text=f"英文 {len(en)} 行 / 中文 {len(zh)} 行")
        if ok:
            self.status.configure(text=f"配对正常：{len(en)} 组；超长行将在生成时自动拆分。")
        else:
            self.status.configure(text="配对未通过：两侧行数必须相同，且每一行都要有内容。")
        if show and not ok:
            messagebox.showwarning("配对检查", "请检查两侧行数是否一致，并确保每一行都有内容。")
        return en, zh, ok

    def reset(self):
        self.en.delete("1.0", "end")
        self.zh.delete("1.0", "end")
        self.en.insert("1.0", SAMPLE_EN)
        self.zh.insert("1.0", SAMPLE_ZH)
        self.check(False)

    def build(self):
        en, zh, ok = self.check(False)
        if not ok:
            self.check(True)
            return
        if len(en) > 6400:
            messagebox.showerror("数量超限", "当前版本最多支持 6400 组配对字形。")
            return
        family = self.family.get().strip()
        if not family:
            messagebox.showerror("缺少字体名称", "请填写字体名称。")
            return
        target = filedialog.asksaveasfilename(
            title="保存生成的字体", defaultextension=".ttf",
            initialfile="TwinLine-Custom.ttf",
            filetypes=[("TrueType Font", "*.ttf")])
        if not target:
            return
        pairs = [{"en": a, "zh": b} for a, b in zip(en, zh)]
        try:
            with tempfile.TemporaryDirectory(prefix="twinline-") as tmp:
                input_path = Path(tmp) / "pairs.json"
                input_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
                cmd = [sys.executable, str(ROOT / "build_font.py"), "--input", str(input_path),
                       "--output", target, "--family", family]
                result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=180)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "未知错误")[-3500:]
                self.status.configure(text="生成失败，请查看错误信息。")
                messagebox.showerror("字体生成失败", detail)
                return
            encoded_path = str(Path(target).with_suffix(".txt"))
            mapping_path = str(Path(target).with_suffix(".mapping.json"))
            self.status.configure(text=f"生成成功：{target}（{len(pairs)} 组）；配套编码文本：{encoded_path}")
            messagebox.showinfo("完成", f"已生成三份配套文件：\n\n字体：{target}\n编码文本：{encoded_path}\n映射表：{mapping_path}\n\n输入 {len(pairs)} 组。超长行已按字宽自动拆分，编码文本会按顺序调用拆分后的字形。将编码文本应用此字体，即可显示英文在上、中文在下。")
        except Exception as exc:
            self.status.configure(text="生成失败。")
            messagebox.showerror("生成失败", str(exc))

if __name__ == "__main__":
    app = tk.Tk()
    FontStudio(app)
    app.mainloop()
