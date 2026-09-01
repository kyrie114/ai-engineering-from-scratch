// 从仓库真实数据生成 20 份 Phase deck(Swiss IKB)。
// 数据来源:README.md(课名/类型/语言/数量/tagline) + 每课 docs/en.md(title/hook/Time) + quiz.json + outputs/。
// 用法:node _gen_phase_decks.mjs [--only=NN]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PHASES } from './_phase_copy.mjs';

const ROOT = 'D:/pycharm_code/AI/ai-engineering-from-scratch';
const OUT = path.join(ROOT, 'project-ppt', 'phases');
const SKILL_TEMPLATE = 'C:/Users/Administrator/.zcode/skills/guizang-ppt-skill/assets/template-swiss.html';
const only = process.argv.find(a => a.startsWith('--only='))?.split('=')[1];

const esc = s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const trimHook = (s, n = 132) => { s = s.trim(); return s.length <= n ? s : s.slice(0, s.lastIndexOf(' ', n)) + ' …'; };
const pad2 = n => String(n).padStart(2, '0');

/* ---------- 1. 解析 README ---------- */
const readme = fs.readFileSync(path.join(ROOT, 'README.md'), 'utf8');
const rowRe = /^\s*\|\s*(\d+)\s*\|\s*\[([^\]]+)\]\((phases\/[^)]+)\)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|/gm;
const phases = [];
{
  const lines = readme.split(/\r?\n/);
  let cur = null;
  for (let i = 0; i < lines.length; i++) {
    const L = lines[i];
    let m = L.match(/^###\s+Phase\s+(\d+):\s*(.+?)\s+`(\d+)\s+lessons`/);
    if (!m) m = L.match(/^<summary><b>Phase\s+(\d+)\s+—\s*(.+?)<\/b>\s*&nbsp;<code>(\d+)\s+lessons<\/code>(?:&nbsp;\s*<em>(.*?)<\/em>)?/);
    if (m) {
      cur = { num: +m[1], title: m[2].trim(), count: +m[3], tagline: (m[4] || '').trim(), lessons: [] };
      phases.push(cur);
      continue;
    }
    if (cur && /^\s*>\s*(.+)/.test(L) && !cur.tagline && !/^>\s*Get your environment/.test(L) === false) {
      // phase 0 的 tagline 在标题下一行
      if (cur.num === 0 && !cur.tagline) cur.tagline = L.replace(/^\s*>\s*/, '').trim();
      continue;
    }
    rowRe.lastIndex = 0;
    const r = rowRe.exec(L);
    if (cur && r) {
      cur.lessons.push({ n: +r[1], title: r[2].trim(), path: r[3].trim(), type: r[4].trim(), lang: r[5].trim() });
    }
  }
  // phase 0 tagline 兜底:标题后第一行 >
  if (phases[0] && !phases[0].tagline) {
    const idx = readme.indexOf('### Phase 0:');
    const seg = readme.slice(idx, idx + 400).match(/>\s*([^\r\n]+)/);
    if (seg) phases[0].tagline = seg[1].trim();
  }
}
if (phases.length !== 20) throw new Error(`expected 20 phases, got ${phases.length}: ${phases.map(p => p.num)}`);

/* ---------- 2. 逐课 enrichment ---------- */
function enrich(lesson, phaseDir) {
  const abs = path.join(ROOT, lesson.path);
  const out = { ...lesson, hook: '', minutes: 0, quiz: 0, artifacts: 0 };
  try {
    const doc = fs.readFileSync(path.join(abs, 'docs', 'en.md'), 'utf8');
    const t = doc.match(/^#\s+(.+)$/m); if (t) out.docTitle = t[1].trim();
    const h = doc.match(/^>\s*(.+)$/m); if (h) out.hook = h[1].trim();
    const timeLine = (doc.match(/\*\*Time:\*\*([^\r\n]*)/) || ['',''])[1];
    const min = timeLine.match(/~?\s*(\d+)\s*minutes?/i);
    const hr = timeLine.match(/~?\s*(\d+)\s*-\s*(\d+)\s*hours?/i) || timeLine.match(/~?\s*(\d+(?:\.\d+)?)\s*hours?/i);
    if (min) out.minutes = +min[1];
    else if (hr) out.minutes = Math.round((hr[2] ? (+hr[1] + +hr[2]) / 2 : +hr[1]) * 60);
  } catch { /* docs 缺失时静默降级 */ }
  try { const q = JSON.parse(fs.readFileSync(path.join(abs, 'quiz.json'), 'utf8')); out.quiz = Array.isArray(q) ? q.length : (q.questions ? q.questions.length : 0); } catch { }
  try { out.artifacts = fs.readdirSync(path.join(abs, 'outputs')).filter(f => !f.startsWith('.')).length; } catch { }
  return out;
}
for (const p of phases) {
  p.lessons = p.lessons.map(l => enrich(l, p));
  p.minutes = p.lessons.reduce((s, l) => s + l.minutes, 0);
  p.hours = Math.round(p.minutes / 60);
  p.quizTotal = p.lessons.reduce((s, l) => s + l.quiz, 0);
  p.artifactTotal = p.lessons.reduce((s, l) => s + l.artifacts, 0);
  const langs = new Set();
  p.lessons.forEach(l => l.lang.split(',').forEach(x => { const t = x.trim(); if (t && t !== '—') langs.add(t); }));
  p.langs = [...langs];
}

/* ---------- 3. 页面模板 ---------- */
const CSS = {
  card: { light: 'background:#f5f5f4;color:var(--text-primary)', grey: 'background:var(--paper);color:var(--text-primary)', dark: 'background:rgba(255,255,255,.06);color:var(--paper)' },
  meta: { light: 'color:var(--text-helper)', grey: 'color:var(--text-helper)', dark: 'color:rgba(255,255,255,.6)' },
};
function chrome(l, r) { return `<div class="chrome-min"><div class="l">${esc(l)}</div><div class="r">${esc(r)}</div></div>`; }
function head(kicker, title, cls = 't-meta') {
  return `<div data-anim="line" style="display:flex;flex-direction:column;gap:1.4vh"><div class="${cls}">${esc(kicker)}</div><h2 class="h-xl-zh">${esc(title)}</h2></div>`;
}

function pCover(cfg, p, idx, total) {
  return `<section class="slide accent" data-animate="hero" data-layout="S01" data-slide-id="p${pad2(cfg.num)}-cover">
  <div class="canvas-card">
    <canvas class="ascii-bg" aria-hidden="true"></canvas>
    ${chrome(`PHASE ${pad2(cfg.num)} · ${p.title.toUpperCase().slice(0, 34)}`, `PHASE GUIDE · ${pad2(idx + 1)} / ${pad2(total)}`)}
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr auto;gap:2.6vh">
      <div data-anim="kicker" class="t-meta" style="color:rgba(255,255,255,.78);letter-spacing:.22em">PHASE ${pad2(cfg.num)} / 20 · ${esc(p.tagline ? '' : '')}AI ENGINEERING FROM SCRATCH</div>
      <div style="margin:auto 0;display:flex;flex-direction:column;gap:2vh">
        <h1 data-anim="title" style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(8.8vw,15vh);line-height:.96;letter-spacing:-.03em;color:#fff">${esc(p.title)}</h1>
        <div data-anim="sub" style="font-family:var(--sans),var(--sans-zh);font-weight:300;font-size:min(2.6vw,4.6vh);letter-spacing:.06em;color:rgba(255,255,255,.85)">${esc(cfg.zh)}</div>
      </div>
      <div data-anim="bottom" style="display:grid;grid-template-rows:auto auto;gap:1.6vh;border-top:1px solid rgba(255,255,255,.22);padding-top:2vh">
        <div data-anim="lead" class="lead" style="max-width:56ch;color:rgba(255,255,255,.86);font-weight:300">${esc(cfg.taglineNote)}</div>
        <div style="display:flex;justify-content:space-between;align-items:end">
          <div class="t-meta" style="color:rgba(255,255,255,.6)">${p.lessons.length} LESSONS · ${esc(p.langs.slice(0, 3).join(' / ').toUpperCase())}${p.langs.length > 3 ? ' +' + (p.langs.length - 3) : ''}</div>
          <div class="t-meta" style="color:rgba(255,255,255,.6)">→ arrow keys</div>
        </div>
      </div>
    </div>
  </div>
</section>`;
}

function pStatement(cfg, p, idx, total) {
  return `<section class="slide" data-animate="statement" data-layout="S09" data-slide-id="p${pad2(cfg.num)}-thesis">
  <div class="canvas-card">
    <span class="dot-mat lg" style="position:absolute;right:0;top:0;width:26vw;height:26vw;color:var(--accent)"></span>
    <span class="ring-mat" style="position:absolute;left:4vw;bottom:14vh;width:14vw;height:14vw;color:var(--grey-3)"></span>
    ${chrome(`PHASE ${pad2(cfg.num)} · 论点`, `${pad2(idx + 1)} / ${pad2(total)}`)}
    <div data-anim="line" class="t-meta" style="letter-spacing:.22em">THE THESIS</div>
    <h1 data-anim="stmt" style="flex:1;align-self:center;margin:auto 0;font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(5vw,8.8vh);line-height:1.18;letter-spacing:-.03em;color:var(--text-primary)">${cfg.hook.map(esc).join('<br/>')}</h1>
    <div data-anim="foot" style="display:flex;justify-content:space-between;align-items:end;gap:3vw;border-top:1px solid var(--border-subtle);padding-top:2vh">
      <p class="t-body-sm" style="max-width:58ch">${esc(trimHook(p.tagline, 110))}</p>
      <span class="t-meta">README · PHASE ${pad2(cfg.num)}</span>
    </div>
  </div>
</section>`;
}

function pKpi(cfg, p, idx, total) {
  const towers = [
    { ic: 'layers', lbl: 'LESSONS', nb: String(p.lessons.length), sub: '节课,docs·code·quiz·outputs 同构', h: 'h-4', acc: ' b-accent' },
    { ic: 'package', lbl: 'ARTIFACTS', nb: String(p.artifactTotal), sub: 'outputs 可复用工件', h: 'h-3', acc: '' },
    { ic: 'clipboard-check', lbl: 'QUIZ QS', nb: p.quizTotal ? String(p.quizTotal) : '—', sub: p.quizTotal ? '随堂测验题总量' : '本阶段暂未配 quiz.json', h: 'h-2', acc: '' },
    { ic: 'languages', lbl: 'LANGUAGES', nb: String(p.langs.length), sub: p.langs.join(' · ').slice(0, 40), h: 'h-1', acc: '' },
  ];
  return `<section class="slide dark" data-animate="measure-up" data-layout="S06" data-slide-id="p${pad2(cfg.num)}-kpi">
  <div class="canvas-card">
    ${chrome(`PHASE ${pad2(cfg.num)} · 体量`, `${pad2(idx + 1)} / ${pad2(total)}`)}
    <div data-anim="line" style="display:grid;grid-template-columns:8fr 4fr;gap:3vw;align-items:end">
      <div style="display:flex;flex-direction:column;gap:1.4vh">
        <div class="t-meta">PHASE ${pad2(cfg.num)} · BY THE NUMBERS</div>
        <h2 class="h-xl-zh">这一阶段有多大</h2>
      </div>
      <p class="t-body-sm" style="padding-bottom:.6vh;color:rgba(255,255,255,.72)">四个数字全部来自仓库实测:目录、docs 标注时间、quiz 与 outputs 逐一累加。</p>
    </div>
    <div class="bar-towers" style="margin-top:4vh">
    ${towers.map(t => `<div class="bar-tower"><div class="cap"><i data-lucide="${t.ic}"></i></div><div class="body-block ${t.h}${t.acc}"><span class="lbl">${t.lbl}</span><span class="nb">${esc(t.nb)}</span><span class="sub">${esc(t.sub)}</span></div></div>`).join('\n    ')}
    </div>
  </div>
</section>`;
}

function pMatrix(cfg, p, idx, total, part, theme) {
  const per = 12, start = part * per;
  const items = p.lessons.slice(start, start + per);
  const remain = p.lessons.length - start - items.length;
  const matrixCount = Math.min(Math.ceil(p.lessons.length / per), 3);
  const card = CSS.card[theme], meta = CSS.meta[theme];
  const cells = items.map(l => `<div style="${card};padding:1.6vh 1vw;display:flex;flex-direction:column;gap:.7vh;min-height:0">
        <div class="t-meta" style="${meta};font-size:14px">L${pad2(l.n)}</div>
        <div style="font-family:var(--sans),var(--sans-zh);font-weight:500;font-size:max(16px,.95vw);line-height:1.3;letter-spacing:-.01em;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden">${esc(l.title)}</div>
        <div class="t-meta" style="${meta};margin-top:auto;font-size:14px">${esc((l.type + ' · ' + l.lang).toUpperCase().slice(0, 26))}</div>
      </div>`).join('\n      ');
  const foot = part === matrixCount - 1 && remain > 0
    ? `\n    <div data-anim="up" style="margin-top:2vh"><div class="t-meta">+ ${remain} MORE · 完整清单见 README 与课程站</div></div>`
    : '';
  return `<section class="slide${theme === 'grey' ? ' grey' : ''}${theme === 'dark' ? ' dark' : ''}" data-animate="matrix-fill" data-layout="S15" data-slide-id="p${pad2(cfg.num)}-map${part + 1}">
  <div class="canvas-card">
    ${chrome(`PHASE ${pad2(cfg.num)} · 课程清单 ${part + 1}/${matrixCount}`, `${pad2(idx + 1)} / ${pad2(total)}`)}
    <div data-anim="line" style="display:flex;flex-direction:column;gap:1.4vh">
      <div class="t-meta">ALL LESSONS · ${start + 1}–${start + items.length} / ${p.lessons.length}</div>
      <h2 class="h-xl-zh">这一阶段的${Math.ceil(p.lessons.length / per) > 1 ? `课单(${part + 1})` : '全部课单'}</h2>
    </div>
    <div data-anim="up" style="flex:1;display:grid;grid-template-columns:repeat(6,1fr);grid-auto-rows:1fr;gap:1.2vh .8vw;margin-top:3.4vh;min-height:0">
      ${cells}
    </div>${foot}
  </div>
</section>`;
}

function pFour(cfg, p, idx, total) {
  const cols = cfg.themes.map(([t, d], i) => `<div style="display:flex;flex-direction:column;gap:1.2vh;border-top:1px solid var(--border-subtle);padding-top:1.6vh">
        <div class="t-meta">— 0${i + 1}</div>
        <h3 style="font-family:var(--sans),var(--sans-zh);font-weight:400;font-size:max(18px,1.35vw);letter-spacing:-.01em">${esc(t)}</h3>
        <p class="t-body-sm">${esc(d)}</p>
      </div>`).join('\n      ');
  return `<section class="slide" data-animate="four-cards" data-layout="S19" data-slide-id="p${pad2(cfg.num)}-themes">
  <div class="canvas-card">
    ${chrome(`PHASE ${pad2(cfg.num)} · 关键主题`, `${pad2(idx + 1)} / ${pad2(total)}`)}
    <div data-anim="line" style="display:flex;flex-direction:column;gap:2.2vh">
      <div style="width:80px;height:2px;background:var(--accent)"></div>
      <div style="display:flex;flex-direction:column;gap:1.2vh">
        <div class="t-meta">FOUR PILLARS OF PHASE ${pad2(cfg.num)}</div>
        <h2 class="h-xl-zh">四块基石</h2>
      </div>
    </div>
    <div data-anim="up" style="flex:1;display:grid;grid-template-columns:repeat(4,1fr);gap:2vw;margin-top:5vh;align-content:start">
      ${cols}
    </div>
  </div>
</section>`;
}

function pDuo(cfg, p, idx, total) {
  const li = arr => arr.map(x => `<li>${esc(x)}</li>`).join('');
  return `<section class="slide dark" data-animate="duo-mirror" data-layout="S08" data-slide-id="p${pad2(cfg.num)}-builduse">
  <div class="canvas-card">
    ${chrome(`PHASE ${pad2(cfg.num)} · 方法论`, `${pad2(idx + 1)} / ${pad2(total)}`)}
    <div data-anim="line" style="display:flex;flex-direction:column;gap:1.4vh">
      <div class="t-meta">THE BUILD IT / USE IT SPLIT · PHASE ${pad2(cfg.num)}</div>
      <h2 class="h-xl-zh">本阶段怎么学</h2>
    </div>
    <div class="duo-compare" style="margin-top:6vh">
      <div class="col">
        <div class="col-tag"><span class="num">01</span>BUILD IT · 手写实现</div>
        <h3 class="col-ttl">${esc(cfg.build.title)}</h3>
        <p class="col-desc">${esc(cfg.build.desc)}</p>
        <ul class="col-list">${li(cfg.build.list)}</ul>
      </div>
      <span class="vrule"></span>
      <div class="col accent">
        <div class="col-tag"><span class="num">02</span>USE IT · 生产级对照</div>
        <h3 class="col-ttl">${esc(cfg.use.title)}</h3>
        <p class="col-desc">${esc(cfg.use.desc)}</p>
        <ul class="col-list">${li(cfg.use.list)}</ul>
      </div>
    </div>
  </div>
</section>`;
}

function pThree(cfg, p, idx, total) {
  const spots = [0, Math.floor(p.lessons.length / 2), p.lessons.length - 1]
    .filter((v, i, a) => a.indexOf(v) === i).slice(0, 3)
    .map(i => p.lessons[i]);
  const cards = spots.map((l, i) => `<div class="card-fill" style="flex:1;padding:2vh 1.6vw;display:grid;grid-template-columns:auto 1fr;gap:1.8vw;align-items:center">
          <div style="font-family:var(--sans);font-weight:200;font-size:min(5vw,8.5vh);line-height:.9;color:var(--accent)">0${i + 1}</div>
          <div style="display:flex;flex-direction:column;gap:.6vh;min-width:0">
            <h3 style="font-family:var(--sans),var(--sans-zh);font-weight:500;font-size:max(17px,1.2vw);letter-spacing:-.01em">L${pad2(l.n)} · ${esc(l.title)}</h3>
            <p class="t-body-sm" style="display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden">${esc(trimHook(l.hook || l.docTitle || l.title, 110))}</p>
          </div>
        </div>`).join('\n        ');
  const list = spots.map(l => `L${pad2(l.n)} ${esc(l.title.toUpperCase().slice(0, 20))}`).join('<br/>');
  return `<section class="slide" data-animate="three-forces" data-layout="S13" data-slide-id="p${pad2(cfg.num)}-spot">
  <div class="canvas-card">
    ${chrome(`PHASE ${pad2(cfg.num)} · 精选三课`, `${pad2(idx + 1)} / ${pad2(total)}`)}
    <div data-anim="up" style="flex:1;display:grid;grid-template-columns:5fr 7fr;gap:1.6vw;margin-top:1vh;min-height:0">
      <div style="background:var(--ink);color:var(--paper);padding:3.2vh 1.8vw;display:flex;flex-direction:column;position:relative;overflow:hidden">
        <span class="dot-mat" style="position:absolute;right:0;bottom:0;width:10vw;height:10vw;color:rgba(255,255,255,.5)"></span>
        <span class="t-cat" style="color:rgba(255,255,255,.72)">LESSON SPOTLIGHTS</span>
        <h2 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(3.4vw,6vh);line-height:1.1;letter-spacing:-.02em;margin-top:1.6vh">先试读三课。</h2>
        <div style="display:flex;flex-direction:column;gap:1.1vh;margin-top:auto;font-family:var(--mono);font-size:max(14px,.8vw);letter-spacing:.06em;color:rgba(255,255,255,.85);position:relative;z-index:1">${list}</div>
      </div>
      <div style="display:flex;flex-direction:column;gap:1.4vh;min-height:0">
        ${cards}
      </div>
    </div>
  </div>
</section>`;
}

function pTimeline(cfg, p, idx, total) {
  const steps = [
    ['STEP 01', '概念直觉', 'docs 六拍讲透问题'],
    ['STEP 02', '手写实现', 'BUILD IT 裸写最小可用版'],
    ['STEP 03', '生产对照', 'USE IT 换生产库验证'],
    ['STEP 04', '随堂测验', 'pre / check / post 查漏'],
    ['STEP 05', '带走工件', 'outputs 直接进工作流'],
  ];
  const nodes = steps.map((s, i) => `<div class="th-node${i % 2 ? ' down' : ' up'}${i === 4 ? ' accent' : ''}"><span class="dot"></span><span class="label"><span class="yr">${s[0]}</span><span class="name">${s[1]}</span><span class="desc">${s[2]}</span></span></div>`).join('\n        ');
  return `<section class="slide grey" data-animate="timeline-walk" data-layout="S11" data-slide-id="p${pad2(cfg.num)}-path">
  <div class="canvas-card">
    ${chrome(`PHASE ${pad2(cfg.num)} · 学习路径`, `${pad2(idx + 1)} / ${pad2(total)}`)}
    <div data-anim="line" style="display:flex;flex-direction:column;gap:1.4vh">
      <div class="t-meta">EVERY LESSON, SAME FIVE STEPS</div>
      <h2 class="h-xl-zh">一课的五步</h2>
    </div>
    <div class="timeline-h" style="margin-top:2vh">
      <div class="tl-row">
        ${nodes}
      </div>
    </div>
    <div class="t-meta" style="margin-top:2vh">BUILD IT → USE IT 是整条课程的方法论脊柱</div>
  </div>
</section>`;
}

function pLedger(cfg, p, idx, total) {
  const rows = [
    [String(p.lessons.length), 'Lessons · 课程', 'docs + code + quiz + outputs 的完整单元', 'layers'],
    [p.hours ? '~' + p.hours.toLocaleString('en-US') : '—', 'Hours · 时长', 'docs 逐课标注合计(毕业项目为项目级时长)', 'clock'],
    [p.quizTotal ? String(p.quizTotal) : '—', 'Quiz questions · 测验', p.quizTotal ? '每课 5-6 题,pre / check / post 三段' : '本阶段课程暂未配 quiz.json', 'clipboard-check'],
    [String(p.artifactTotal), 'Artifacts · 工件', 'outputs/ 可复用:prompt · skill · agent · MCP', 'package'],
    [String(p.langs.length), 'Languages · 语言', p.langs.join(' · ') || '—', 'languages'],
  ];
  const rowHtml = rows.map(r => `<div class="ledger-row" style="display:grid;grid-template-columns:minmax(9vw,auto) 1fr auto;gap:3vw;align-items:center;border-bottom:1px solid rgba(255,255,255,.16);padding:1.2vh 0">
        <span class="ledger-num" style="font-family:var(--sans);font-weight:200;font-size:min(4.6vw,6.8vh);line-height:1;letter-spacing:-.03em;font-feature-settings:'tnum'">${esc(r[0])}</span>
        <span class="ledger-label" style="display:flex;flex-direction:column;gap:.3vh"><span class="t-body-emp">${esc(r[1])}</span><span class="t-body-sm">${esc(r[2])}</span></span>
        <span class="ledger-icon" style="opacity:.75;display:flex"><i data-lucide="${r[3]}" style="width:1.8vw;height:1.8vw"></i></span>
      </div>`).join('\n      ');
  return `<section class="slide dark" data-animate="stacked-ledger" data-layout="S20" data-slide-id="p${pad2(cfg.num)}-assets">
  <div class="canvas-card">
    ${chrome(`PHASE ${pad2(cfg.num)} · 阶段资产`, `${pad2(idx + 1)} / ${pad2(total)}`)}
    <div data-anim="line" style="display:flex;flex-direction:column;gap:1.2vh">
      <div class="t-meta">PHASE ${pad2(cfg.num)} · WHAT YOU GET</div>
      <h2 class="h-xl-zh">学完带走多少</h2>
    </div>
    <div data-anim="ledger" style="flex:1;display:flex;flex-direction:column;justify-content:center;margin-top:1vh;min-height:0">
      ${rowHtml}
    </div>
  </div>
</section>`;
}

function pManifesto(cfg, p, idx, total) {
  return `<section class="slide" data-animate="manifesto" data-layout="S12" data-slide-id="p${pad2(cfg.num)}-creed">
  <div class="canvas-card">
    ${chrome(`PHASE ${pad2(cfg.num)} · 阶段信条`, `${pad2(idx + 1)} / ${pad2(total)}`)}
    <div style="flex:1;display:grid;grid-template-columns:7fr 5fr;gap:4vw;align-items:start;padding-top:2vh">
      <div data-anim="line" style="display:flex;flex-direction:column;gap:2vh">
        <div class="t-cat accent">PHASE ${pad2(cfg.num)} · ${esc(cfg.zh.toUpperCase())}</div>
        <h2 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(4.6vw,8.2vh);line-height:1.14;letter-spacing:-.03em">${cfg.manifesto.map(esc).join('<br/>')}</h2>
      </div>
      <div style="display:flex;flex-direction:column;gap:2vh;padding-top:1.2vw">
        <span class="ring-mat" style="display:inline-block;width:6vw;height:6vw;color:var(--grey-3)"></span>
        <p class="t-body" style="max-width:38ch">${esc(cfg.manifestoNote)}</p>
        <p class="t-body-sm">结构由仓库审计脚本把关:${p.lessons.length} 课同构,翻开任何一课都不会迷路。</p>
      </div>
    </div>
    <div data-anim="up" style="margin:4vh -5vw -4.4vh;background:var(--ink);color:var(--paper);padding:3vh 5vw;display:flex;align-items:center;justify-content:space-between;gap:2vw">
      <span style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(2.6vw,4.6vh);letter-spacing:-.01em">BUILD IT · USE IT · <span style="color:var(--accent-bright)">SHIP IT</span></span>
      <span class="t-meta" style="color:rgba(255,255,255,.6)">PHASE ${pad2(cfg.num)} / 20</span>
    </div>
  </div>
</section>`;
}

function pClosing(cfg, p, idx, total, next) {
  const nextSlug = PHASES[cfg.num + 1]?.slug;
  const takeaways = [
    ['01', '你造了什么', cfg.buildSummary, false],
    ['02', '你带走什么', `${p.artifactTotal} 件 outputs 工件 + ${p.quizTotal} 道随堂测验题,全部可复查。`, false],
    ['03', next ? `下一站 · Phase ${pad2(next.num)}` : '下一站', next ? `${next.title}(${next.lessons.length} 课),deck 见 phases/${nextSlug}/index.html` : '回到课程站,挑一条深度构建线继续打穿。', true],
  ];
  const items = takeaways.map(([n, t, d, acc]) => `<div style="display:grid;grid-template-columns:auto 1fr;gap:2vw;align-items:start;padding:2.6vh 0;border-top:1px solid var(--border-subtle)${acc ? ';border-bottom:2px solid var(--accent)' : ''}">
            <div style="font-family:var(--sans);font-weight:200;font-size:min(4.4vw,7.8vh);line-height:.9;color:${acc ? 'var(--accent)' : 'var(--text-primary)'}">${n}</div>
            <div>
              <h3 style="font-family:var(--sans),var(--sans-zh);font-weight:400;font-size:max(18px,1.55vw);line-height:1.2;letter-spacing:-.015em;color:${acc ? 'var(--accent)' : 'var(--text-primary)'};margin-bottom:1vh">${esc(t)}</h3>
              <p style="font-family:var(--sans),var(--sans-zh);font-size:max(16px,.94vw);line-height:1.6;color:var(--text-secondary);font-weight:400">${esc(d)}</p>
            </div>
          </div>`).join('\n          ');
  return `<section class="slide split" data-animate="split-statement" data-layout="S10" data-slide-id="p${pad2(cfg.num)}-closing">
  <div class="canvas-card">
    <div class="split-half">
      <div class="half b-accent" style="padding:5.6vh 3.6vw 4.4vh;justify-content:space-between;position:relative;overflow:hidden">
        <canvas class="ascii-bg" aria-hidden="true"></canvas>
        <div class="chrome-min" style="margin-bottom:0;position:relative;z-index:1">
          <div class="l">${pad2(idx + 1)} / ${pad2(total)}</div>
          <div class="r">CLOSING</div>
        </div>
        <div data-anim="manifesto" style="display:flex;flex-direction:column;gap:2vh;position:relative;z-index:1">
          <div class="t-meta" style="color:rgba(255,255,255,.78);letter-spacing:.22em;margin-bottom:1.6vh">PHASE ${pad2(cfg.num)} · ${esc(cfg.zh.toUpperCase())}</div>
          <h2 style="font-family:var(--sans),var(--sans-zh);font-size:min(6.8vw,12vh);line-height:1.02;letter-spacing:-.025em;font-weight:200;color:#fff">${cfg.closingBig.map(esc).join('<br/>')}</h2>
        </div>
        <div data-anim="signature" style="display:flex;justify-content:space-between;align-items:end;border-top:1px solid rgba(255,255,255,.22);padding-top:2vh;position:relative;z-index:1">
          <div class="t-meta" style="color:rgba(255,255,255,.62)">${esc(trimHook(p.tagline, 46))}</div>
          <div class="t-meta" style="color:rgba(255,255,255,.62)">26.08.30</div>
        </div>
      </div>
      <div class="half" style="padding:5.6vh 3.6vw 4.4vh;justify-content:space-between">
        <div class="chrome-min"><div class="l">TAKEAWAYS</div><div class="r">03 STEPS</div></div>
        <div data-anim="rules" class="takeaway-list" style="display:flex;flex-direction:column;gap:0">
          ${items}
        </div>
        <div data-anim="foot" class="t-meta" style="color:var(--text-helper);text-align:right">→ PHASE ${pad2(cfg.num)} 完</div>
      </div>
    </div>
  </div>
</section>`;
}

/* ---------- 4. 演讲备注 ---------- */
function notes(cfg, p, pagePlan, total, next) {
  const base = `Phase ${pad2(cfg.num)} ${cfg.zh}(${p.lessons.length} 课)`;
  const per = {
    cover: { title: `封面 · Phase ${pad2(cfg.num)} ${p.title}`, m: .4, purpose: '报出阶段名与体量,建立本份 deck 的预期', talk: [`${base},是 20 个阶段里的第 ${cfg.num + 1} 站`, `一句话定位:${cfg.hook.join('')}`, '预告结构:论点 → 数据 → 课单 → 方法 → 工件'], trans: '先给一个论点' },
    thesis: { title: `论点 · ${cfg.hook.join('')}`, m: .5, purpose: '用一句话立住本阶段的世界观', talk: [`README 原话:${trimHook(p.tagline, 60)}`, cfg.taglineNote], trans: '用数字校准体量感' },
    kpi: { title: '体量 · KPI 塔', m: .6, purpose: '四个实测数字建立体量感', talk: [`${p.lessons.length} 课,docs 标注合计约 ${p.hours} 小时`, `quiz 共 ${p.quizTotal} 题、outputs 工件 ${p.artifactTotal} 件`, '数字全部来自仓库目录实测'], trans: '按顺序过一遍课单' },
    map: { title: '课单', m: .8, purpose: '扫一遍本阶段所有课程标题', talk: ['不逐课展开,挑标题里重复出现的关键词讲', '提示观众:同款课单在课程站可点进每一课', '后面有三课试读'], trans: '把课单压成四个关键主题' },
    themes: { title: '四块基石', m: .6, purpose: '给课单建立记忆骨架', talk: cfg.themes.map(t => `${t[0]}:${t[1]}`), trans: '讲这一阶段独有的学习方法' },
    builduse: { title: 'Build It / Use It', m: .8, purpose: '方法论在本阶段的具体化', talk: [`BUILD IT:${cfg.build.desc}`, `USE IT:${cfg.use.desc}`, '这是整条课程的脊柱,不只是本阶段'], trans: '挑三课实读' },
    spot: { title: '精选三课', m: .8, purpose: '用真实课名与官方 hook 展示课程质感', talk: spots => { }, trans: '给出通用学习路径' },
    path: { title: '一课的五步', m: .5, purpose: '给出可复用的单课学习法', talk: ['概念 → 手写 → 对照 → 测验 → 工件', '每课 quiz 分 pre/check/post 三段', '工件是学完当天就能用的'], trans: '汇总阶段资产' },
    assets: { title: '阶段资产', m: .6, purpose: '回答「学完我拿到什么」', talk: [`${p.lessons.length} 课 · 约 ${p.hours} 小时`, `${p.quizTotal} 题 + ${p.artifactTotal} 件工件`, `语言覆盖:${p.langs.join('、') || '—'}`], trans: '收束成一句信条' },
    creed: { title: `信条 · ${cfg.manifesto.join('')}`, m: .5, purpose: '留下本阶段的记忆锚点', talk: [cfg.manifestoNote, '重复一遍底部三个动词:Build · Use · Ship'], trans: '给出行动建议' },
    closing: { title: '收束 · 三步', m: .6, purpose: '把本阶段收成行动', talk: [cfg.buildSummary, `工件 ${p.artifactTotal} 件、测验 ${p.quizTotal} 题可复查`, next ? `下一站 Phase ${pad2(next.num)} ${next.title}` : '最后一阶段,回到课程站选深度线'], trans: '结束本份 deck' },
  };
  return pagePlan.filter(pl => pl.key !== 'mapExtra').map(pl => {
    const n = pl.key === 'map' ? { ...per.map, title: `课单 ${pl.part + 1}` } : per[pl.key];
    return {
      id: `p${pad2(cfg.num)}-${pl.key}${pl.key === 'map' ? pl.part + 1 : ''}`,
      title: n.title, section: cfg.zh, minutes: n.m,
      purpose: n.purpose,
      talk: pl.key === 'spot'
        ? [0, Math.floor(p.lessons.length / 2), p.lessons.length - 1].filter((v, i, a) => a.indexOf(v) === i).slice(0, 3).map(i => { const l = p.lessons[i]; return `L${pad2(l.n)} ${l.title}:${trimHook(l.hook || '', 70)}`; })
        : n.talk,
      transition: n.trans,
    };
  });
}

/* ---------- 5. 组装 ---------- */
const template = fs.readFileSync(SKILL_TEMPLATE, 'utf8');
const startMark = '<!-- ============ 示例:第 1 页';
const endMark = '</div>\r\n\r\n<div id="nav"></div>';

let made = 0;
for (const cfg of PHASES) {
  if (only && String(cfg.num) !== only) continue;
  const p = phases.find(x => x.num === cfg.num);
  if (!p) throw new Error('phase data missing: ' + cfg.num);
  const next = phases.find(x => x.num === cfg.num + 1) || null;
  const nL = p.lessons.length;
  const matrixCount = Math.min(Math.ceil(nL / 12), nL > 24 ? 3 : nL > 12 ? 2 : 1);
  const plan = [
    { key: 'cover' }, { key: 'thesis' }, { key: 'kpi' },
    ...Array.from({ length: matrixCount }, (_, part) => ({ key: 'map', part })),
    { key: 'themes' }, { key: 'builduse' }, { key: 'spot' }, { key: 'path' }, { key: 'assets' }, { key: 'creed' }, { key: 'closing' },
  ];
  const total = plan.length;
  const secHtml = plan.map((pl, idx) => {
    if (pl.key === 'cover') return pCover(cfg, p, idx, total);
    if (pl.key === 'thesis') return pStatement(cfg, p, idx, total);
    if (pl.key === 'kpi') return pKpi(cfg, p, idx, total);
    if (pl.key === 'map') return pMatrix(cfg, p, idx, total, pl.part, pl.part === 1 ? 'grey' : pl.part === 2 ? 'dark' : 'light');
    if (pl.key === 'themes') return pFour(cfg, p, idx, total);
    if (pl.key === 'builduse') return pDuo(cfg, p, idx, total);
    if (pl.key === 'spot') return pThree(cfg, p, idx, total);
    if (pl.key === 'path') return pTimeline(cfg, p, idx, total);
    if (pl.key === 'assets') return pLedger(cfg, p, idx, total);
    if (pl.key === 'creed') return pManifesto(cfg, p, idx, total);
    if (pl.key === 'closing') return pClosing(cfg, p, idx, total, next);
  }).join('\n\n');

  let html = template;
  const start = html.indexOf(startMark), end = html.indexOf(endMark);
  if (start < 0 || end < 0) throw new Error('template markers not found');
  html = html.slice(0, start) + secHtml + '\r\n\r\n' + html.slice(end);
  const nStart = html.indexOf('const SPEAKER_NOTES = ['), nEnd = html.indexOf('window.__SPEAKER_NOTES__ = SPEAKER_NOTES;');
  html = html.slice(0, nStart) + 'const SPEAKER_NOTES = ' + JSON.stringify(notes(cfg, p, plan, total, next)) + ';\r\n' + html.slice(nEnd);
  html = html.replace('<title>[必填] 替换为 PPT 标题 · Deck Title</title>', `<title>Phase ${pad2(cfg.num)} · ${esc(p.title)} · AI Engineering from Scratch</title>`);
  html = html.replace("await import('./assets/motion.min.js')", "await import('../../assets/motion.min.js')");
  if (html.includes('[必填]')) throw new Error('placeholder left in phase ' + cfg.num);

  const dir = path.join(OUT, cfg.slug);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'index.html'), html, 'utf8');
  made++;
  console.log(`phase ${pad2(cfg.num)} ${cfg.slug}: ${total} pages · ${nL} lessons · ${p.hours}h · quiz ${p.quizTotal} · artifacts ${p.artifactTotal}`);
}
console.log('generated decks:', made);
