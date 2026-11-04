# TwinLine V2 — 双层中英组合字体

**V2 是真实 TrueType 字体生成器。** 每个字形是一组固定文本对：英文在上、对应中文在下。不是把预览图伪装成字体。

## 直接使用：可视化字体工作台

新增 `font_studio.py`，不需要手动编辑 JSON：

1. 安装 Python 3、Tkinter、fontTools、fontconfig，以及 Lato / Noto Sans CJK 字体。
2. 启动 `python font_studio.py`。
3. 左侧粘贴英文，每行一条；右侧粘贴对应中文，每行一条。
4. 点击“检查配对”，确认两侧行数一致、每行均有内容。
5. 点击“生成 TTF 字体”，选择保存位置。

工作台会自动生成临时配对数据、调用字体引擎、按实际字宽自动拆分超长行、检查字形轮廓，并输出三份配套文件：TTF 字体、同名 `.mapping.json` 映射表、同名 `.txt` 编码文本。临时 JSON 会自动清理，不需要手工创建或修改项目文件。**要实际显示双层字形，必须把生成的 `.txt` 编码文本应用刚生成的 TTF 字体**；不能直接把原始英文/中文普通文本贴进字体后就自动叠排。

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

## 普通文本直接排版（无需 PUA）

打开 [TwinLine V2 排版器](specimen/index.html)。可以直接粘贴普通英文和中文，选择四种配对方式：

- **逐行配对**：英文第 N 行对应中文第 N 行，由用户显式指定对应关系。\n- **逐短语配对**：每行是一组句子，使用 `|` 标记短语边界；两侧同一行、同一序号的短语构成独立配对单元。程序按真实 LF/CRLF/CR 换行解析并保留空行位置；校验行数和每行短语数量，空行、空短语或数量不一致都会整次停止输出，不会过滤空行造成后续错位。
- **按句子配对**：分别识别英文和中文句子，按顺序建立句子对；句子数量不同则停止输出。
- **按段落配对**：先按空行分段，再校验每一对段落的句子数量；任一段不一致则停止输出。

只有通过对应关系校验后，排版器才会测量两种字体的实际宽度并计算视觉断行。窗口宽度或字号变化时重新排版。句子内部的行切分是词边界/宽度层面的视觉近似，不会假装理解翻译语义；短语模式支持在同一行内用 `|` 建立多个显式短语对，例如 `take | a closer look | at the issue` 对应 `仔细 | 看一看 | 这个问题`。程序能保证结构位置不串位，但不能验证两个短语在语义上是否真的互译；这一步仍需人工确认或接入双语对齐模型。

两侧数量不一致时会明确提示，不会把后续文本静默错位。可调整字号、层间距，导出独立 HTML 文件，或打印/保存为 PDF。导出文件可直接在浏览器打开，不需要安装字体或转换私用区字符。

## 自动语义对齐（本地模型）

新增 `tools/semantic_align.py`：使用 AWESOME-align 的 multilingual BERT 词级对齐模型，不再按句子序号或字符长度猜测对应关系。模型在本机运行；首次运行会从 Hugging Face 下载模型权重（默认 `bert-base-multilingual-cased`），之后可使用缓存模型离线推理。

安装（建议 Python 3.9–3.11，PyTorch 按机器 CPU/CUDA 环境选择）：

```bash
python -m pip install -r tools/requirements-semantic-align.txt
```

执行一组英中句子后，会生成 `alignment.json`。打开 `specimen/index.html`，点击“导入模型对齐 JSON”，选择该文件；在候选列表中勾选不重叠、语序一致的短语，再点“应用勾选的短语配对”。编辑器会载入短语模式，仍需人工检查后才能导出。

执行一组英中句子：

```bash
python tools/semantic_align.py \
  --en "take a closer look at the issue" \
  --zh "仔细看一看这个问题" \
  --output alignment.json
```

输出 JSON 包含英文词、中文词的分词结果、模型给出的词级对应链接及链接概率，以及基于词级链接一致性提取的短语候选、跨度、覆盖率、对齐分数和高/中/低分档。

候选提取遵循短语一致性约束：候选中文跨度不能包含连接到候选英文跨度之外的词。短语候选是模型建议，不是自动翻译，也不保证每个候选都语义正确。分数是模型内部对齐启发式分数，不是经过校准的“翻译正确概率”；习语、省略、倒装和跨句语义仍需人工核对。模型权重约数百 MB，CPU 推理可能较慢；可用 `--no-cuda` 强制 CPU，或用 `--model` 指向本地模型目录。
## 在线排版预览

浏览器排版器处理任意普通文本；独立 TTF 则将有限的配对内容固化为私用区字形，两者用途不同。

## 技术限制（必须明确）

- 一个组合字形代表一对固定文本；这不是自动翻译器，也不会把任意普通英文自动翻译成中文。
- 新文本通过工作台输入后生成新的组合字形、映射表和 PUA 编码文本；将该编码文本设置为生成字体即可显示。
- TTF 字形是固定轮廓，不能在系统字体层自动理解任意两段普通文本。生成器会按源字体实际字宽分别拆分英文和中文，并按顺序输出多个连续组合字形，避免单字形过宽；拆分点是排版断点，不代表机器理解了翻译语义。
- 复杂脚本 shaping、双向文字与段落自动换行由上层排版器处理。
- 派生字体受 Lato 与 Noto CJK 的 SIL Open Font License 约束，发布时需保留许可证。

## 自动构建

GitHub Actions 会在生成器、样例或测试变化时运行语法检查、配对输入单元测试、真实字体构建和轮廓校验，并上传 TTF、映射 JSON 与 PUA 编码 TXT。


## 整段自动语义对齐（第二阶段）

新增 `tools/align_document.py`。输入完整英文段落和中文段落，不要求两侧句子数量相同。流程：

1. 使用专门用于跨语言语义相似度的 multilingual Sentence-Transformers 句向量计算跨语言相似度。
2. 使用单调动态规划，寻找全局最优句子对应，支持 1:1、1:2、2:1；明显不相关或多余的句子会标记为未匹配。
3. 将匹配后的句组批量交给 AWESOME-align 做词级对齐，再提取短语候选。
4. 输出结构与原浏览器候选导入格式兼容的 JSON；未匹配句子单独列出。

运行：

```bash
python tools/align_document.py \
  --en "The first sentence explains the idea. A second sentence adds detail." \
  --zh "第一句话解释了这个想法。第二句话补充了细节。" \
  --output document-alignment.json
```

注意：当前句子匹配保持原文顺序，不会任意重排；句向量相似度和对齐分数都不是翻译正确概率。句子层可以识别不同数量及 1:2/2:1 分组，但真实语料准确率仍需基准测试。句级模型默认使用 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，词级模型使用 `bert-base-multilingual-cased`；两者均在本地运行；首次下载模型后可使用缓存。

## 英文—简体中文真实语料准确率基准（V4）

基准语料改为 **WMT News Commentary v16**。它是新闻评论领域的英文—中文平行语料，官方以单个 gzip TSV 文件分发；每一行自身包含一条英文句子和对应中文句子，中间用 Tab 分隔。下载器直接流式解压同一文件，不再把两个独立文本文件按行号拼接，也不依赖 PyArrow。

官方语料地址：

\`https://data.statmt.org/news-commentary/v16/training/news-commentary-v16.en-zh.tsv.gz\`

输出仅保留英文源句与简体中文目标句：使用 OpenCC 检测并排除会发生繁转简变化的目标句，再排除含常见书面粤语标记的句子。粤语规则是启发式过滤，不是完整的方言分类器；简体字形也不等于绝对普通话。

\`\`\`bash
python -m pip install opencc-python-reimplemented
python tools/download_parallel_corpus.py --output data/news-commentary-en-zh.tsv --limit 10000
python tools/benchmark_parallel.py \\
  --input data/news-commentary-en-zh.tsv \\
  --models sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,sentence-transformers/LaBSE \\
  --block-size 20 \\
  --output reports/news-commentary-en-zh-benchmark.json
\`\`\`

\`--limit 10000\` 限制写出的有效配对数量，但下载器仍会读完整个 gzip 文件并统计所有过滤数量，以便验证压缩流完整性。下载器会生成同名 \`.manifest.json\`，记录扫描数、有效数、写入数及各类过滤数。

基准比较 MiniLM 与 LaBSE 两个句向量模型，只在校准集上扫描相似度阈值和 DP gap/merge 惩罚参数，再在保留的连续测试块上报告 exact-match accuracy、precision、recall、F1 和错配案例。这个留出集来自同一训练语料文件，不是 WMT 官方 test split；源文件没有稳定的文档 ID，因此不能声称文档级完全独立。结果只代表 News Commentary v16 语料上的句子对齐表现，不代表翻译质量或所有领域的普遍准确率。

运行基础回归测试：

```bash
python -m unittest discover -s tests -v
```

