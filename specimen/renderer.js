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
      if (!line || measure(candidate) <= maxWidth) {
        if (measure(candidate) <= maxWidth) {
          line = candidate;
          continue;
        }
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

  function splitOneLine(lines, maxWidth, measure) {
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
      const distance = Math.abs(measure(accumulated) - target);
      if (accumulated.trim() && (value.slice(accumulated.length).trim()) && distance < nearest) {
        nearest = distance;
        cut = accumulated.length;
      }
    }
    if (cut <= 0 || cut >= value.length) {
      const chars = graphemes(value);
      let left = "";
      let right = "";
      let leftWidth = 0;
      let split = 0;
      for (let i = 1; i < chars.length; i++) {
        left += chars[i - 1];
        const width = measure(left);
        const distance = Math.abs(width - target);
        if (distance < nearest) {
          nearest = distance;
          split = i;
          leftWidth = width;
        }
      }
      if (!split || !leftWidth) return false;
      left = chars.slice(0, split).join("");
      right = chars.slice(split).join("");
      lines.splice(best, 1, left, right);
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
      if (!splitOneLine(top, width, measureEn)) break;
    }
    guard = 0;
    while (bottom.length < target && guard++ < 10000) {
      if (!splitOneLine(bottom, width, measureZh)) break;
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
    return value.replace(/\r/g, "").split("\n");
  }

  function render(options) {
    const { english, chinese, mode, container, enFont, zhFont, enSize, zhSize, gap } = options;
    const enBlocks = mode === "paragraphs" ? paragraphBlocks(english) : lineBlocks(english);
    const zhBlocks = mode === "paragraphs" ? paragraphBlocks(chinese) : lineBlocks(chinese);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const width = Math.max(1, container.clientWidth);
    const measure = (font) => text => {
      ctx.font = font;
      return ctx.measureText(text).width;
    };
    const measureEn = measure("500 " + enSize + "px " + enFont);
    const measureZh = measure("400 " + zhSize + "px " + zhFont);
    container.replaceChildren();
    container.style.setProperty("--en-size", enSize + "px");
    container.style.setProperty("--zh-size", zhSize + "px");
    container.style.setProperty("--gap", gap + "px");

    const count = Math.max(enBlocks.length, zhBlocks.length);
    for (let i = 0; i < count; i++) {
      const en = enBlocks[i] || "";
      const zh = zhBlocks[i] || "";
      const group = document.createElement("section");
      group.className = "pair";
      group.dataset.pair = String(i + 1);
      const rows = wrapPair(en, zh, width, measureEn, measureZh);
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
    }
    return { en: enBlocks, zh: zhBlocks, rows: count };
  }

  const api = { wrapText, wrapPair, paragraphBlocks, lineBlocks, render };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.TwinLineRenderer = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
