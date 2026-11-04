const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");
(async () => {
 const browser = await chromium.launch({headless:true});
 try {
  const page = await browser.newPage({viewport:{width:390,height:844}});
  const errors=[]; page.on("pageerror",e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.resolve(__dirname,"../specimen/index.html")).href);
  await page.evaluate(()=>document.fonts.ready);
  const alignmentFixture={
   model:"fixture-model",
   phrase_candidates:[
    {source:"take a",target:"仔细",source_span:[0,1],target_span:[0,0],score:0.88,confidence_band:"high"},
    {source:"closer look",target:"看一看",source_span:[2,3],target_span:[1,1],score:0.86,confidence_band:"high"}
   ]
  };
  await page.locator("#alignmentFile").setInputFiles({
   name:"alignment.json",mimeType:"application/json",
   buffer:Buffer.from(JSON.stringify(alignmentFixture))
  });
  await page.waitForFunction(()=>!document.querySelector("#alignmentReview").hidden);
  assert.equal(await page.locator("#alignmentCandidates input").count(),2);
  await page.locator("#alignmentCandidates input").nth(0).check();
  await page.locator("#alignmentCandidates input").nth(1).check();
  await page.locator("#applyAlignment").click();
  assert.equal(await page.locator("#english").inputValue(),"take a | closer look");
  assert.equal(await page.locator("#chinese").inputValue(),"仔细 | 看一看");
  assert.equal(await page.locator("#mode").inputValue(),"phrases");
  await page.locator("#english").fill("take | a closer look | at the issue");
  await page.locator("#chinese").fill("仔细 | 看一看 | 这个问题");
  await page.locator("#mode").selectOption("phrases");
  await page.waitForFunction(()=>document.querySelectorAll("#preview .pair").length===3);
  let phraseData=await page.evaluate(()=>Array.from(document.querySelectorAll("#preview .pair")).map(p=>({
   en:Array.from(p.querySelectorAll(".en")).map(x=>x.textContent).join("").replace(/\s/g,""),
   zh:Array.from(p.querySelectorAll(".zh")).map(x=>x.textContent).join("")
  })));
  assert.deepEqual(phraseData,[
   {en:"take",zh:"仔细"},{en:"acloserlook",zh:"看一看"},{en:"attheissue",zh:"这个问题"}
  ]);
  await page.locator("#english").fill("one | two\nthree | four");
  await page.locator("#chinese").fill("一 | 二\n三 | 四");
  await page.waitForFunction(()=>document.querySelectorAll("#preview .pair").length===4);
  const multilinePairs=await page.evaluate(()=>Array.from(document.querySelectorAll("#preview .pair")).map(p=>[
   p.querySelector(".en").textContent,p.querySelector(".zh").textContent
  ]));
  assert.deepEqual(multilinePairs,[["one","一"],["two","二"],["three","三"],["four","四"]]);
  await page.locator("#english").fill("one | two\n\nthree | four");
  await page.locator("#chinese").fill("一 | 二\n\n三 | 四");
  await page.waitForFunction(()=>document.querySelector("#status").textContent.includes("短语为空"));
  assert.equal(await page.locator("#preview .pair").count(),0);
  await page.locator("#english").fill("take | a closer look | at the issue");
  await page.locator("#chinese").fill("仔细 | 看一看");
  await page.waitForFunction(()=>document.querySelector("#status").textContent.includes("句子行数相同"));
  assert.equal(await page.locator("#preview .pair").count(),0);
  await page.locator("#chinese").fill("仔细 | 看一看 | 这个问题");
  await page.locator("#english").fill("The first sentence is here. The second sentence follows!");
  await page.locator("#chinese").fill("第一句话在这里。第二句话紧随其后！");
  await page.locator("#mode").selectOption("sentences");
  await page.waitForFunction(()=>document.querySelectorAll("#preview .pair").length===2);
  let data=await page.evaluate(()=>Array.from(document.querySelectorAll("#preview .pair")).map(p=>({
   en:Array.from(p.querySelectorAll(".en")).map(x=>x.textContent).join("").replace(/\s/g,""),
   zh:Array.from(p.querySelectorAll(".zh")).map(x=>x.textContent).join("")
  })));
  assert.equal(data.length,2);
  assert.equal(data[0].en,"Thefirstsentenceishere.");
  assert.equal(data[0].zh,"第一句话在这里。");
  assert.equal(data[1].en,"Thesecondsentencefollows!");
  assert.equal(data[1].zh,"第二句话紧随其后！");
  await page.locator("#chinese").fill("只有一句中文。");
  await page.waitForFunction(()=>document.querySelector("#status").textContent.includes("数量不一致"));
  assert.equal(await page.locator("#preview .pair").count(),0);
  assert.equal(await page.locator("#preview .alignment-error").count(),1);
  await page.locator("#english").fill("First sentence. Second sentence.\n\nAnother paragraph.");
  await page.locator("#chinese").fill("第一句。第二句。第三句。\n\n另一个段落。");
  await page.locator("#mode").selectOption("paragraphs");
  await page.waitForFunction(()=>document.querySelector("#status").textContent.includes("第 1 段句子数量不一致"));
  assert.equal(await page.locator("#preview .pair").count(),0);
  await page.locator("#english").fill("The first sentence is here. The second sentence follows!");
  await page.locator("#chinese").fill("第一句话在这里。第二句话紧随其后！");
  await page.locator("#mode").selectOption("sentences");
  await page.waitForFunction(()=>document.querySelectorAll("#preview .pair").length===2);
  await page.setViewportSize({width:900,height:844});
  await page.waitForTimeout(100);
  const overflow=await page.evaluate(()=>Array.from(document.querySelectorAll("#preview .en,#preview .zh")).some(n=>n.scrollWidth>n.clientWidth+1));
  assert.equal(overflow,false);
  assert.deepEqual(errors,[]);
  console.log("Chromium alignment/responsive tests passed");
 } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
