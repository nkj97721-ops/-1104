const assert = require("node:assert/strict");
const {
  wrapText, wrapPair, splitOneLine, paragraphBlocks, lineBlocks,
  splitSentences, preparePairs, phraseLines
} = require("../specimen/renderer.js");

const measure = text => Array.from(text).length * 10;
const en = "The quick brown fox jumps over the lazy dog.";
const zh = "敏捷的棕色狐狸跳过了那只懒狗。";
const rows = wrapPair(en, zh, 70, measure, measure);
assert.ok(rows.length > 1);
assert.equal(rows.map(row => row.en).join("").replace(/\s/g, ""), en.replace(/\s/g, ""));
assert.equal(rows.map(row => row.zh).join(""), zh);
assert.deepEqual(paragraphBlocks("a\nb\n\nc\nd"), ["a b", "c d"]);
assert.deepEqual(lineBlocks("a\nb"), ["a", "b"]);

const clauseLines = ["First clause, second clause continues"];
assert.equal(splitOneLine(clauseLines, measure), true);
assert.deepEqual(clauseLines, ["First clause,", "second clause continues"]);
const chineseClauseLines = ["第一部分，第二部分继续"];
assert.equal(splitOneLine(chineseClauseLines, measure), true);
assert.deepEqual(chineseClauseLines, ["第一部分，", "第二部分继续"]);

assert.deepEqual(phraseLines("take | a closer look\\nat | the issue"), [
  { line: 1, parts: ["take", "a closer look"] },
  { line: 2, parts: ["at", "the issue"] }
]);
assert.deepEqual(phraseLines("take | a closer look\\r\\nat | the issue"), [
  { line: 1, parts: ["take", "a closer look"] },
  { line: 2, parts: ["at", "the issue"] }
]);
assert.deepEqual(phraseLines("one | two\\n\\nthree"), [
  { line: 1, parts: ["one", "two"] },
  { line: 2, parts: [""] },
  { line: 3, parts: ["three"] }
]);
const phrasePairs = preparePairs(
  "take | a closer look | at the issue",
  "仔细 | 看一看 | 这个问题",
  "phrases"
);
assert.equal(phrasePairs.errors.length, 0);
assert.deepEqual(phrasePairs.pairs.map(x => [x.en, x.zh]), [
  ["take", "仔细"], ["a closer look", "看一看"], ["at the issue", "这个问题"]
]);
assert.ok(phrasePairs.pairs.every(x => x.source === "phrase"));
const phraseMismatch = preparePairs(
  "take | a closer look",
  "仔细 | 看一看 | 这个问题",
  "phrases"
);
assert.equal(phraseMismatch.pairs.length, 0);
assert.match(phraseMismatch.errors[0], /短语数量不一致/);
const phraseLineMismatch = preparePairs("one | two\\nthree", "一 | 二", "phrases");
assert.equal(phraseLineMismatch.pairs.length, 0);
assert.match(phraseLineMismatch.errors[0], /句子行数相同/);
const phraseBlankLine = preparePairs("one | two\\n\\nthree", "一 | 二\\n\\n三", "phrases");
assert.equal(phraseBlankLine.pairs.length, 0);
assert.ok(phraseBlankLine.errors.some(message => /短语为空/.test(message)));
const phraseTrailingDelimiter = preparePairs("one |", "一 | 二", "phrases");
assert.equal(phraseTrailingDelimiter.pairs.length, 0);
assert.ok(phraseTrailingDelimiter.errors.some(message => /短语为空/.test(message)));

const sentenceEn = "The first sentence is here. The second sentence follows!";
const sentenceZh = "第一句话在这里。第二句话紧随其后！";
assert.equal(splitSentences(sentenceEn).length, 2);
assert.equal(splitSentences(sentenceZh).length, 2);

const sentencePairs = preparePairs(sentenceEn, sentenceZh, "sentences");
assert.equal(sentencePairs.errors.length, 0);
assert.deepEqual(sentencePairs.pairs.map(x => x.en), [
  "The first sentence is here.", "The second sentence follows!"
]);
assert.deepEqual(sentencePairs.pairs.map(x => x.zh), [
  "第一句话在这里。", "第二句话紧随其后！"
]);

const mismatch = preparePairs(
  "The first sentence. The second sentence.",
  "只有一句中文。",
  "sentences"
);
assert.equal(mismatch.pairs.length, 0);
assert.equal(mismatch.errors.length, 1);
assert.match(mismatch.errors[0], /数量不一致/);

const paragraphMismatch = preparePairs(
  "First sentence. Second sentence.\n\nAnother paragraph.",
  "第一句。第二句。第三句。\n\n另一个段落。",
  "paragraphs"
);
assert.equal(paragraphMismatch.pairs.length, 0);
assert.equal(paragraphMismatch.errors.length, 1);
assert.match(paragraphMismatch.errors[0], /第 1 段句子数量不一致/);

const emptyInput = preparePairs("", "", "sentences");
assert.equal(emptyInput.pairs.length, 0);
assert.equal(emptyInput.errors.length, 1);
assert.match(emptyInput.errors[0], /都必须填写/);

const lineMismatch = preparePairs("English line one.\nEnglish line two.", "只有一行中文。", "lines");
assert.equal(lineMismatch.pairs.length, 0);
assert.equal(lineMismatch.errors.length, 1);

const goodParagraphs = preparePairs(
  "First sentence. Second sentence.\n\nAnother paragraph.",
  "第一句。第二句。\n\n另一个段落。",
  "paragraphs"
);
assert.equal(goodParagraphs.errors.length, 0);
assert.equal(goodParagraphs.pairs.length, 3);
assert.equal(goodParagraphs.pairs[2].paragraph, 2);

console.log("semantic alignment tests passed");
