# TwinLine V2 — 双层中英组合字体

**V2 是真实 TrueType 字体生成器。** 每个字形是一组固定文本对：英文在上、对应中文在下。不是把预览图伪装成字体。

## 直接使用：可视化字体工作台

新增 `font_studio.py`，不需要手动编辑 JSON：

1. 安装 Python 3、Tkinter、fontTools、fontconfig，以及 Lato / Noto Sans CJK 字体。
2. 启动 `python font_studio.py`。
3. 左侧粘贴英文，每行一条；右侧粘贴对应中文，每行一条。
4. 点击“检查配对”，确认两侧行数一致、每行均有内容。
5. 点击“生成 TTF 字体”，选择保存位置。

工作台会自动生成临时配对数据、调用字体引擎、检查字形轮廓，并输出三份配套文件：TTF 字体、同名 `.mapping.json` 映射表、同名 `.txt` 编码文本。临时 JSON 会自动清理，不需要手工创建或修改项目文件。**要实际显示双层字形，必须把生成的 `.txt` 编码文本应用刚生成的 TTF 字体**；不能直接把原始英文/中文普通文本贴进字体后就自动叠排。

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

输出包括 TTF、映射表 JSON 和可直接套用字体的 `.txt` 编码文本。每一对文本被编码为一个 Unicode 私用区字符（U+E000 起），所以必须同时使用对应 TTF 和编码文本；普通英文/中文字符本身不会自动触发组合字形。

## 在线排版预览

打开 [specimen/index.html](specimen/index.html)，输入英文与中文逐行配对，调整两层字号和间距，并导出双层排版 HTML。浏览器预览用于校对版式；真正的 TTF 由本地字体引擎生成。

## 技术限制（必须明确）

- 一个组合字形代表一对固定文本；这不是自动翻译器，也不会把任意普通英文自动翻译成中文。
- 新文本通过工作台输入后生成新的组合字形、映射表和 PUA 编码文本；将该编码文本设置为生成字体即可显示。
- 当前组合字形是固定轮廓，字形内部英文与中文不会像普通文本那样独立换行；长句会横向变宽。单字形宽度超过 TrueType 16 位 advance 上限时会明确报错，需把两侧拆成对应的较短行。
- 复杂脚本 shaping、双向文字与段落自动换行由上层排版器处理。
- 派生字体受 Lato 与 Noto CJK 的 SIL Open Font License 约束，发布时需保留许可证。

## 自动构建

GitHub Actions 会在生成器、样例或测试变化时运行语法检查、配对输入单元测试、真实字体构建和轮廓校验，并上传 TTF、映射 JSON 与 PUA 编码 TXT。
