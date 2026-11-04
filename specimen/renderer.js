(function (root) {
  "use strict";

  function graphemes(text) {
    if (typeof Intl !== "undefined" && Intl.Segmenter) {
      return Array.from(new Intl.Segmenter(undefined, { granularity: "grapheme" }).segment(text), x => x.segment);
    }
    return Array.from(text);
  }

  function wordTokens(text) {
    if (typeof Intl !== "undefined" && Intl.Segmenter) {
      return Array.from(new Intl.Segmenter(undefined, { granularity: "word" }).segment(text), x => x.segment);
    }
    return graphemes(text);
  }

  function splitLongToken(token, maxWidth, measure) {
    const chars = graphemes(token);
    const result = [];
    let part = "";
    for (const char of chars) {
      const next = part + char;
      if (part && measure(next) > maxWidth) {
        result.push(part);
        part = char;
      } else {
        part = next;
      }
    }
    if (part) result.push(part);
    return result;
  }

  function wrapText(text, maxWidth, measure) {
    if (!text) return [""];
    const lines = [];
    let line = "";
    for (const token of wordTokens(text)) {
      const candidate = line + token;
      if (measure(candidate) <= maxWidth) {
        line = candidate;
        continue;
      }
      if (line.trim()) lines.push(line.trimEnd());
      line = "";
      const cleanToken = token.trimStart();
      if (measure(cleanToken) <= maxWidth) {
        line = cleanToken;
      } else {
        const fragments = splitLongToken(cleanToken, maxWidth, measure);
        for (let i = 0; i < fragments.length - 1; i++) lines.push(fragments[i]);
        line = fragments[fragments.length - 1] || "";
      }
    }
    if (line || !lines.length) lines.push(line.trimEnd());
    return lines;
  }

  function splitOneLine(lines, measure) {
    let best = -1;
    let bestWidth = -1;
    for (let i = 0; i < lines.length; i++) {
      const width = measure(lines[i]);
      if (graphemes(lines[i]).length > 1 && width > bestWidth) {
        best = i;
        bestWidth = width;
      }
    }
    if (best < 0) return false;

    const value = lines[best];
    const tokens = wordTokens(value);
    let cut = -1;
    let accumulated = "";
    const target = measure(value) / 2;
    let nearest = Infinity;
    for (let i = 1; i < tokens.length; i++) {
      accumulated += tokens[i - 1];
      const left = accumulated.trimEnd();
      const right = value.slice(accumulated.length).trimStart();
      if (!left || !right) continue;
      const distance = Math.abs(measure(left) - target);
      if (distance < nearest) {
        nearest = distance;
        cut = accumulated.length;
      }
    }

    if (cut <= 0 || cut >= value.length) {
      const chars = graphemes(value);
      let left = "";
      let split = 0;
      nearest = Infinity;
      for (let i = 1; i < chars.length; i++) {
        left += chars[i - 1];
        const distance = Math.abs(measure(left) - target);
        if (distance < nearest) {
          nearest = distance;
          split = i;
        }
      }
      if (!split) return false;
      lines.splice(best, 1, chars.slice(0, split).join(""), chars.slice(split).join(""));
      return true;
    }

    lines.splice(best, 1, value.slice(0, cut).trimEnd(), value.slice(cut).trimStart());
    return true;
  }

  function wrapPair(en, zh, width, measureEn, measureZh) {
    const top = wrapText(en, width, measureEn);
    const bottom = wrapText(zh, width, measureZh);
    const target = Math.max(top.length, bottom.length);
    let guard = 0;
    while (top.length < target && guard++ < 10000) {
      if (!splitOneLine(top, measureEn)) break;
    }
    guard = 0;
    while (bottom.length < target && guard++ < 10000) {
      if (!splitOneLine(bottom, measureZh)) break;
    }
    while (top.length < target) top.push("");
    while (bottom.length < target) bottom.push("");
    return top.map((line, i) => ({ en: line, zh: bottom[i] || "" }));
  }

  function paragraphBlocks(value) {
    return value.replace(/\r/g, "").trim().split(/\n\s*\n+/)
      .map(block => block.replace(/\n+/g, " ").trim()).filter(Boolean);
  }

  function lineBlocks(value) {
    return value.replace(/\r/g, "").split("\n").map(line => line.trim());
  }

  function splitSentences(value) {
    const text = value.replace(/\r/g, "").replace(/\n+/g, " ").trim();
    if (!text) return [];
    if (typeof Intl !== "undefined" && Intl.Segmenter) {
      return Array.from(new Intl.Segmenter(undefined, { granularity: "sentence" }).segment(text),
        item => item.segment.trim()).filter(Boolean);
    }
    return (text.match(/[^.!?。！？；;]+[.!?。！？；;]?/g) || [text])
      .map(sentence => sentence.trim()).filter(Boolean);
  }

  function preparePairs(english, chinese, mode) {
    const errors = [];
    const pairs = [];
    if (mode === "lines") {
      const en = lineBlocks(english);
      const zh = lineBlocks(chinese);
      if (en.length !== zh.length) {
        errors.push("逐行模式要求英文和中文行数相同：英文 " + en.length + " 行，中文 " + zh.length + " 行。");
        return { pairs, errors };
      }
      en.forEach((value, i) => {
        if (!value || !zh[i]) errors.push("第 " + (i + 1) + " 行为空，请补齐对应内容。");
        else pairs.push({ en: value, zh: zh[i], source: "line", index: i + 1 });
      });
      return { pairs, errors };
    }

    if (mode === "sentences") {
      const en = splitSentences(english);
      const zh = splitSentences(chinese);
      if (en.length !== zh.length) {
        errors.push("句子数量不一致：英文 " + en.length + " 句，中文 " + zh.length + " 句。请调整句子边界，禁止自动错配。");
        return { pairs, errors };
      }
      en.forEach((value, i) => pairs.push({ en: value, zh: zh[i], source: "sentence", index: i + 1 }));
      return { pairs, errors };
    }

    const enParagraphs = paragraphBlocks(english);
    const zhParagraphs = paragraphBlocks(chinese);
    if (enParagraphs.length !== zhParagraphs.length) {
      errors.push("段落数量不一致：英文 " + enParagraphs.length + " 段，中文 " + zhParagraphs.length + " 段。请先对应段落。");
      return { pairs, errors };
    }
    enParagraphs.forEach((paragraph, paragraphIndex) => {
      const enSentences = splitSentences(paragraph);
      const zhSentences = splitSentences(zhParagraphs[paragraphIndex]);
      if (enSentences.length !== zhSentences.length) {
        errors.push("第 " + (paragraphIndex + 1) + " 段句子数量不一致：英文 " +
          enSentences.length + " 句，中文 " + zhSentences.length + " 句。该段不会输出。");
        return;
      }
      enSentences.forEach((value, sentenceIndex) => {
        pairs.push({
          en: value, zh: zhSentences[sentenceIndex],
          source: "paragraph", paragraph: paragraphIndex + 1, index: sentenceIndex + 1
        });
      });
    });
    if (errors.length) pairs.length = 0;
    return { pairs, errors };
  }

  function render(options) {
    const { english, chinese, mode, container, enFont, zhFont, enSize, zhSize, gap } = options;
    const prepared = preparePairs(english, chinese, mode);
    if (!prepared.errors.length && !prepared.pairs.length) {
      prepared.errors.push("没有可排版的配对内容，请填写英文和对应中文。");
    }
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const width = Math.max(1, container.clientWidth - 2);
    const measure = (font, letterSpacing) => text => {
      ctx.font = font;
      return ctx.measureText(text).width + Math.max(0, graphemes(text).length - 1) * letterSpacing;
    };
    const measureEn = measure("500 " + enSize + "px " + enFont, enSize * 0.005);
    const measureZh = measure("400 " + zhSize + "px " + zhFont, zhSize * 0.015);
    container.replaceChildren();
    container.style.setProperty("--en-size", enSize + "px");
    container.style.setProperty("--zh-size", zhSize + "px");
    container.style.setProperty("--gap", gap + "px");

    if (prepared.errors.length) {
      const warning = document.createElement("div");
      warning.className = "alignment-error";
      warning.setAttribute("role", "alert");
      prepared.errors.forEach(message => {
        const item = document.createElement("p");
        item.textContent = message;
        warning.append(item);
      });
      container.append(warning);
      return { valid: false, errors: prepared.errors, en: [], zh: [], pairs: [], rows: 0 };
    }

    let rowCount = 0;
    prepared.pairs.forEach((pair, i) => {
      const group = document.createElement("section");
      group.className = "pair";
      group.dataset.pair = String(i + 1);
      group.dataset.source = pair.source;
      const rows = wrapPair(pair.en, pair.zh, width, measureEn, measureZh);
      rowCount += rows.length;
      rows.forEach((row, index) => {
        const line = document.createElement("div");
        line.className = "line-pair";
        line.dataset.line = String(index + 1);
        const top = document.createElement("p");
        top.className = "en";
        top.textContent = row.en;
        const bottom = document.createElement("p");
        bottom.className = "zh";
        bottom.textContent = row.zh;
        line.append(top, bottom);
        group.append(line);
      });
      container.append(group);
    });

    return {
      valid: true, errors: [], pairs: prepared.pairs,
      en: prepared.pairs.map(pair => pair.en),
      zh: prepared.pairs.map(pair => pair.zh),
      rows: rowCount
    };
  }

  const api = {
    wrapText, wrapPair, paragraphBlocks, lineBlocks,
    splitSentences, preparePairs, render
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.TwinLineRenderer = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
