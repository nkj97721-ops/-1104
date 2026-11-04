#!/usr/bin/env python3
"""Build document-specific composite glyphs: English above its Chinese line."""
import argparse, json, os, subprocess
from pathlib import Path
from fontTools.ttLib import TTFont, TTCollection
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.cu2quPen import Cu2QuPen

UPM, PUA_START = 2048, 0xE000

def fc_match(pattern):
    path = subprocess.run(['fc-match','-f','%{file}',pattern],check=True,text=True,capture_output=True).stdout.strip()
    if not path or not os.path.isfile(path): raise RuntimeError('Font not found: '+pattern)
    return path

def load_cjk(path):
    if path.lower().endswith(('.ttc','.otc')):
        coll=TTCollection(path)
        for f in coll.fonts:
            names=[n.toUnicode() for n in f['name'].names if n.nameID==1]
            if any('CJK SC' in n for n in names): return f
        raise RuntimeError('Simplified Chinese face missing')
    return TTFont(path)

def source(font):
    cmap={}
    for t in font['cmap'].tables:
        if t.isUnicode(): cmap.update(t.cmap)
    return font.getGlyphSet(),cmap,font['hmtx'].metrics

def draw_text(text,gs,cmap,metrics,pen,scale,baseline,missing):
    x=0
    for ch in text:
        name=cmap.get(ord(ch))
        if name is None:
            missing.add(ch); name=cmap.get(0x25A1) or cmap.get(ord('?'))
        if name is None: x+=600*scale; continue
        adv,_=metrics[name]
        target=TransformPen(pen,(scale,0,0,scale,x,baseline))
        gs[name].draw(Cu2QuPen(target,max_err=1.0))
        x+=adv*scale
    return x

def build(pairs,output,family):
    latin_path=fc_match('Lato:style=Medium')
    cjk_path=fc_match('Noto Sans CJK SC:style=Regular')
    lgs,lcmap,lmetrics=source(TTFont(latin_path))
    cgs,ccmap,cmetrics=source(load_cjk(cjk_path))
    order=['.notdef']; cmap={}; glyf={'.notdef':TTGlyphPen(None).glyph()}; hmtx={'.notdef':(UPM,0)}
    missing=set(); mapping=[]
    for i,pair in enumerate(pairs):
        name=f'pair{i:04d}'; pen=TTGlyphPen(None)
        ew=draw_text(pair['en'],lgs,lcmap,lmetrics,pen,.48,1120,missing)
        zw=draw_text(pair['zh'],cgs,ccmap,cmetrics,pen,.48,110,missing)
        glyph=pen.glyph(); glyf[name]=glyph
        hmtx[name]=(max(UPM,int(max(ew,zw)+100)),0); order.append(name)
        cp=PUA_START+i
        if cp>0xF8FF: raise ValueError('Too many pairs for BMP PUA')
        cmap[cp]=name
        mapping.append({'index':i+1,'codepoint':f'U+{cp:04X}','en':pair['en'],'zh':pair['zh']})
    fb=FontBuilder(UPM,isTTF=True); fb.setupGlyphOrder(order); fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyf); fb.setupHorizontalMetrics(hmtx); fb.setupHorizontalHeader(ascent=2048,descent=-180)
    fb.setupNameTable({'familyName':family,'styleName':'Regular','uniqueFontIdentifier':family+' Regular 1.0','fullName':family+' Regular','psName':family.replace(' ','')+'-Regular','version':'Version 1.000'})
    fb.setupOS2(sTypoAscender=2048,sTypoDescender=-180,usWinAscent=2048,usWinDescent=180,sxHeight=1050,sCapHeight=1400,usWeightClass=500,usWidthClass=5)
    fb.setupPost(); fb.setupMaxp(); font=fb.font; font['head'].created=font['head'].modified=0
    Path(output).parent.mkdir(parents=True,exist_ok=True); font.save(output)
    Path(output).with_suffix('.mapping.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
    check=TTFont(output); gs=check.getGlyphSet(); bad=[]
    for name in order[1:]:
        bp=BoundsPen(gs); gs[name].draw(bp)
        if bp.bounds and (bp.bounds[1]<-30 or bp.bounds[3]>2048): bad.append((name,bp.bounds))
    if bad: raise RuntimeError('Vertical overflow: '+repr(bad[:5]))
    if len(check.getBestCmap())!=len(pairs): raise RuntimeError('cmap count mismatch')
    if missing: raise RuntimeError('Missing source glyphs: '+repr(sorted(missing)))
    print(json.dumps({'font':output,'bytes':os.path.getsize(output),'pairs':len(pairs),'vertical_bounds':'PASS','missing_glyphs':[],'mapping':str(Path(output).with_suffix('.mapping.json'))},ensure_ascii=False,indent=2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='samples/pairs.json')
    ap.add_argument('--output',default='dist/TwinLine-V2-Sample.ttf')
    ap.add_argument('--family',default='TwinLine V2 Sample')
    a=ap.parse_args(); pairs=json.loads(Path(a.input).read_text(encoding='utf-8'))
    if not pairs or any(not isinstance(x.get('en'),str) or not isinstance(x.get('zh'),str) for x in pairs): raise SystemExit('Expected non-empty [{en,zh}] JSON array')
    build(pairs,a.output,a.family)
if __name__=='__main__': main()
