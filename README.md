# TwinLine V2 — 双层中英组合字体

**V2 是真实 TrueType 字体生成器。** 每个字形是一组固定文本对：英文在上、对应中文在下。不是把预览图伪装成字体。

## 直接使用：可视化字体工作台

新增 `font_studio.py`，不需要手动编辑 JSON：

1. 安装 Python 3、Tkinter、fontTools、fontconfig，以及 Lato / Noto Sans CJK 字体。
2. 启动 `python font_studio.py`。
3. 左侧粘贴英文，每行一条；右侧粘贴对应中文，每行一条。
4. 点击“检查配对”，确认两侧行数一致、每行均有内容。
5. 点击“生成 TTF 字体”，选择保存位置。

工作台会自动生成临时配对数据、调用字体引擎、检查字形轮廓并输出 TTF 和同名 mapping JSON。临时 JSON 会自动清理，不需要手工创建或修改项目文件。

## 安装依赖

Ubuntu / Debian：

```bash
sudo apt-get update
sudo apt-get install -y python3-tk fonts-lato fonts-noto-cjk fontconfig
python3 -m pip install fonttools
```

macOS / Windows：安装 Python 3、fontTools、Lato 和 Noto Sans CJK SC；确保 `fc-match` 可用（Windows 可使用 Fontconfig 环境）。

## 命令行方式

```bash
python build_font.py --input samples/pairs.json --output dist/My-TwinLine.ttf --family "My TwinLine"
```

输出包括 TTF 与映射表 JSON。映射表将每一对文本关联到 Unicode 私用区码位 U+E000 起始位置。

## 在线排版预览

打开 [specimen/index.html](specimen/index.html)，输入英文与中文逐行配对，调整两层字号和间距，并导出双层排版 HTML。浏览器预览用于校对版式；真正的 TTF 由本地字体引擎生成。

## 技术限制（必须明确）

- 一个组合字形代表一对固定文本；这不是自动翻译器，也不会把任意普通英文自动翻译成中文。
- 新文本通过工作台输入后生成新的组合字形和映射表。
- 当前组合字形是固定轮廓，字形内部英文与中文不会像普通文本那样独立换行；长句会横向变宽。
- 复杂脚本 shaping、双向文字与段落自动换行由上层排版器处理。
- 派生字体受 Lato 与 Noto CJK 的 SIL Open Font License 约束，发布时需保留许可证。

## 自动构建

GitHub Actions 会在生成器或样例数据变化时构建并验证字体，并上传 TTF 与映射 JSON 构建产物。
