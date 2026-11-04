# TwinLine V2 — 双层中英组合字体

**V2 已生成真实 TrueType 字体文件。** 每个字形是一个固定文本对：英文在上、对应中文在下。不是图片，而是可安装、可嵌入文档的 TTF 轮廓字形。

## 成品与验证

- 本轮已本地构建并通过字体解析、PUA 映射数量、字形垂直边界检查。
- [排版编辑器](specimen/index.html)：编辑英文/中文逐行配对并预览。
- [字体生成器](build_font.py)：从配对文本生成 TTF 和映射表。
- [样例文本对](samples/pairs.json)：四组演示数据。
- [CI 构建记录](https://github.com/nkj97721-ops/-1104/actions)：每次修改生成器或样例都会自动构建字体并上传构建产物。

## 使用方式

这是**文档专用组合字形字体**。样例使用 Unicode 私用区 U+E000 起始码位。普通英文字母不会自动变成翻译后的双层字形；排版程序需把每一对英文/中文替换为映射表中的对应私用区字符，再应用本字体。

例：第一组文本对对应 U+E000。把该字符放入文档并应用 TwinLine V2 字体，就会显示一枚上下双层组合字形。

## 自定义生成

编辑 `samples/pairs.json`，每项填写 `en` 与 `zh`，然后执行：

```bash
sudo apt-get install fonts-lato fonts-noto-cjk fontconfig
python -m pip install fonttools
python build_font.py --input samples/pairs.json --output dist/My-TwinLine.ttf --family "My TwinLine"
```

脚本将拉丁与简体中文字形轮廓合成为一个字形，分配 PUA 编码，输出 TTF 与 JSON 映射表，并执行结构及轮廓边界检查。

## 限制

- 一个组合字形代表一对固定文本；新增配对需重新生成字体与映射。
- 这是固定版式/文档嵌入方案，不是能自动翻译任意输入的通用系统字体。
- 复杂脚本 shaping、双向文字与自动换行由上层排版器处理。
- 派生字体受 Lato 与 Noto CJK 的 SIL Open Font License 约束，随成品附带许可证。
