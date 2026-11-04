# 技术实现与验证

## 目标模型

每一对英文/中文行合成为一个独立轮廓字形：英文置上，中文置下，两行左边界对齐。一个字形代表一个明确文本对，不依赖字体猜测翻译。

## 已实现

1. 双层排版编辑器：`specimen/index.html`，按行配对、调整两层字号/间距、行数不一致时告警。
2. TTF 生成器：`build_font.py`，使用 fontTools 读取 Lato Medium 与 Noto Sans CJK SC 字形轮廓，组合成 TrueType glyf。
3. PUA 映射：从 U+E000 起为每个文本对分配码位，并输出同名 `.mapping.json`。
4. 配套编码输出：每组文本分配一个 U+E000 起始的私用区码位，同时输出可直接套用生成字体的 `.txt` 文件与完整映射表。
5. 自动构建：GitHub Actions 安装字体源与 fontTools，运行输入校验测试，构建 TTF、映射文件和编码文本并上传 artifact。

## 本地验证结果

- 输出格式：TrueType TTF。
- 样例文本对：4 组。
- TTF 文件大小：25,588 bytes。
- cmap：样例私用区映射数量与输入一致。
- 编码文本：每个配对单元对应一个私用区字符，字符顺序与映射表一致。
- 缺失源字形：0。
- 所有组合字形垂直边界均处于设计范围内。
- fontTools 重新解析成功；fontconfig 能识别字体；Pillow 实际渲染预览成功。

## 重要边界

普通 TTF/OTF 不会理解英文语义，也不会自动生成中文译文。任意英文与中文的语义配对必须由用户或上层翻译程序提供。生成字体只包含配对组合的私用区字形，因此需要同时使用对应的编码文本；直接输入普通英文/中文不会自动合成双层字形。本项目的字体把已经配对的两行固化为单个组合字形，适用于固定文档、海报、标题或嵌入式排版。

## 自定义构建

修改 `samples/pairs.json` 后执行：

```bash
sudo apt-get install fonts-lato fonts-noto-cjk fontconfig
python -m pip install fonttools
python build_font.py --input samples/pairs.json --output dist/My-TwinLine.ttf --family "My TwinLine"
```

输出 TTF 与 JSON 映射表。字体及其派生轮廓遵守上游 Lato 和 Noto CJK SIL OFL 许可。
