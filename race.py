"""
Build the race: every agent running the SAME episode, one small board each.

An earlier version drew all the agents overlaid on a single board. Don't go
back to it: each agent has its *own* prey, so twenty agents means forty objects
on forty-nine cells, and the one thing a viewer needs to track -- who is
chasing whom -- is exactly what gets lost.

Small multiples instead. Two objects per board, so each board is readable at a
glance, and the boards re-sort themselves live so the race is still a race.
The prey is the same colour and shape on every board; only the hunter takes
the agent's colour.

Writes one self-contained HTML file. No network, no dependencies, opens
straight off disk onto a projector.
"""

import json

from core import GRID, WALLS

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hunt Arena &mdash; the race</title>
<style>
  :root { color-scheme: dark; --prey: #ffc244; }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: #0d1117; color: #e6edf3;
    font: 15px/1.4 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    height: 100vh; display: flex; flex-direction: column; overflow: hidden;
  }
  .topbar {
    display: flex; align-items: center; gap: 22px; flex-wrap: wrap;
    padding: 14px 20px; border-bottom: 1px solid #21262d; flex: 0 0 auto;
  }
  h1 { margin: 0; font-size: 19px; font-weight: 650; letter-spacing: -.01em; }
  .legend { display: flex; align-items: center; gap: 16px; font-size: 14px; color: #adbac7; }
  .legend b { color: #e6edf3; font-weight: 600; }
  .swatch { display: inline-flex; align-items: center; gap: 7px; }
  .dot { width: 15px; height: 15px; border-radius: 50%; background: #7ee787; }
  .dia {
    width: 13px; height: 13px; background: var(--prey);
    transform: rotate(45deg); border-radius: 2px;
  }
  .controls { margin-left: auto; display: flex; align-items: center; gap: 10px; }
  button, select {
    font: inherit; font-size: 14px; color: #e6edf3; background: #21262d;
    border: 1px solid #30363d; border-radius: 8px; padding: 6px 13px; cursor: pointer;
  }
  button:hover, select:hover { background: #30363d; }
  input[type=range] { width: 110px; vertical-align: middle; }
  .step { font-variant-numeric: tabular-nums; color: #8b949e; font-size: 14px; min-width: 96px; }

  .grid {
    flex: 1 1 auto; overflow-y: auto; padding: 16px 20px 24px;
    display: grid; gap: 14px;
    grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    align-content: start;
  }
  .card {
    background: #161b22; border: 1px solid #21262d; border-radius: 12px;
    padding: 9px 10px 10px; transition: border-color .12s, box-shadow .12s;
    cursor: pointer; min-width: 0;
  }
  .card.flash { border-color: var(--prey); box-shadow: 0 0 0 2px rgba(255,194,68,.28); }
  .card.bot { background: #14181e; }
  .hdr { display: flex; align-items: center; gap: 7px; margin-bottom: 7px; min-width: 0; }
  .rank {
    font-variant-numeric: tabular-nums; font-size: 12px; color: #8b949e;
    min-width: 20px;
  }
  .chip { width: 11px; height: 11px; border-radius: 3px; flex: 0 0 auto; }
  .nm {
    font-size: 13px; font-weight: 600; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; flex: 1 1 auto; min-width: 0;
  }
  .bot .nm { color: #9aa4ae; font-style: italic; font-weight: 500; }
  .sc {
    font-variant-numeric: tabular-nums; font-size: 18px; font-weight: 700;
    flex: 0 0 auto;
  }
  canvas { width: 100%; height: auto; display: block; border-radius: 7px; }
  .avg { margin-top: 5px; font-size: 11px; color: #6e7681; font-variant-numeric: tabular-nums; }

  .overlay {
    position: fixed; inset: 0; background: rgba(6,9,13,.93); display: none;
    align-items: center; justify-content: center; flex-direction: column; gap: 16px;
    z-index: 10;
  }
  .overlay.on { display: flex; }
  .overlay canvas { width: min(74vh, 88vw); border-radius: 14px; }
  .otitle { font-size: 24px; font-weight: 700; display: flex; align-items: center; gap: 12px; }
  .ohint { color: #8b949e; font-size: 14px; }
</style>
</head>
<body>
<div class="topbar">
  <h1>Hunt Arena</h1>
  <div class="legend">
    <span class="swatch"><span class="dot"></span><b>hunter</b> &mdash; the agent</span>
    <span class="swatch"><span class="dia"></span><b>prey</b> &mdash; the script</span>
    <span>same episode on every board</span>
  </div>
  <div class="controls">
    <select id="show">
      <option value="0">show all</option>
      <option value="12">top 12</option>
      <option value="6">top 6</option>
    </select>
    <button id="play">Pause</button>
    <button id="restart">Restart</button>
    <label class="step">speed <input id="speed" type="range" min="1" max="30" value="8"></label>
    <span class="step" id="stepout">step 0</span>
  </div>
</div>
<div class="grid" id="grid"></div>
<div class="overlay" id="overlay">
  <div class="otitle" id="otitle"></div>
  <canvas id="ocanvas" width="560" height="560"></canvas>
  <div class="ohint">click anywhere to close</div>
</div>
<script>
const DATA = /*DATA*/;
const G = DATA.grid, WALLS = new Set(DATA.walls.map(w => w[0] + "," + w[1]));
const A = DATA.agents, T = A[0].hunter.length;
const PREY = "#ffc244";

let hue = 0;
A.forEach(a => {
  a.color = a.bot ? "#8b949e" : `hsl(${Math.round(hue)} 70% 62%)`;
  if (!a.bot) hue += 360 / Math.max(1, A.filter(x => !x.bot).length);
  a.cum = [0];
  for (let t = 0; t < a.caught.length; t++) a.cum.push(a.cum[t] + (a.caught[t] ? 1 : 0));
});

function board(cx, size, a, t, big) {
  const C = size / G;
  cx.clearRect(0, 0, size, size);
  cx.fillStyle = "#0f141a"; cx.fillRect(0, 0, size, size);
  for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) {
    cx.fillStyle = WALLS.has(r + "," + c) ? "#30373f" : "#1a2029";
    cx.fillRect(c * C + 1, r * C + 1, C - 2, C - 2);
  }
  const p = a.prey[t], h = a.hunter[t];
  // prey: amber diamond, identical on every board
  const px = p[1] * C + C / 2, py = p[0] * C + C / 2, pr = C * (big ? .26 : .28);
  cx.save(); cx.translate(px, py); cx.rotate(Math.PI / 4);
  cx.fillStyle = PREY; cx.fillRect(-pr, -pr, pr * 2, pr * 2);
  cx.restore();
  // hunter: filled disc in the agent's colour, with a dark rim so it reads
  // even when it lands on top of the prey
  const hx = h[1] * C + C / 2, hy = h[0] * C + C / 2;
  cx.beginPath(); cx.arc(hx, hy, C * (big ? .30 : .32), 0, 7);
  cx.fillStyle = a.color; cx.fill();
  cx.lineWidth = big ? 3 : 2; cx.strokeStyle = "#0d1117"; cx.stroke();
}

const grid = document.getElementById("grid");
const cards = A.map((a, i) => {
  const el = document.createElement("div");
  el.className = "card" + (a.bot ? " bot" : "");
  el.innerHTML =
    `<div class="hdr"><span class="rank"></span>`
    + `<span class="chip" style="background:${a.color}"></span>`
    + `<span class="nm"></span><span class="sc">0</span></div>`
    + `<canvas width="210" height="210"></canvas>`
    + `<div class="avg">${a.per_ep.toFixed(1)} / ep over ${DATA.episodes} episodes</div>`;
  // textContent, not innerHTML: the name comes from an uploaded filename.
  el.querySelector(".nm").textContent = a.name;
  el.onclick = () => openFocus(i);
  grid.appendChild(el);
  const cv = el.querySelector("canvas");
  return { el, cx: cv.getContext("2d"), rank: el.querySelector(".rank"),
           sc: el.querySelector(".sc") };
});

let limit = 0;
function render(t) {
  const order = A.map((a, i) => [a.cum[t], i]).sort((x, y) => y[0] - x[0] || x[1] - y[1]);
  order.forEach(([v, i], pos) => {
    const c = cards[i];
    c.el.style.order = pos;
    c.el.style.display = (limit && pos >= limit) ? "none" : "";
    c.rank.textContent = pos + 1;
    c.sc.textContent = v;
    c.el.classList.toggle("flash", t > 0 && A[i].caught[t - 1]);
    if (!limit || pos < limit) board(c.cx, 210, A[i], t, false);
  });
  if (focused >= 0) {
    board(ocx, 560, A[focused], t, true);
    otitle.innerHTML = `<span class="chip" style="background:${A[focused].color};`
      + `width:18px;height:18px;border-radius:5px"></span>`
      + `<span class="onm"></span>`
      + ` &mdash; ${A[focused].cum[t]} catches`;
    otitle.querySelector(".onm").textContent = " " + A[focused].name;
  }
  document.getElementById("stepout").textContent = `step ${t} / ${T - 1}`;
}

const overlay = document.getElementById("overlay");
const ocx = document.getElementById("ocanvas").getContext("2d");
const otitle = document.getElementById("otitle");
let focused = -1;
function openFocus(i) { focused = i; overlay.classList.add("on"); render(t); }
overlay.onclick = () => { focused = -1; overlay.classList.remove("on"); };

let t = 0, playing = true, fps = 8, acc = 0, last = performance.now();
const playBtn = document.getElementById("play");
function loop(now) {
  const dt = now - last; last = now;
  if (playing) {
    acc += dt;
    while (acc > 1000 / fps) { acc -= 1000 / fps; t = Math.min(t + 1, T - 1); }
    if (t >= T - 1) { playing = false; playBtn.textContent = "Play"; }
  }
  render(t);
  requestAnimationFrame(loop);
}
playBtn.onclick = e => {
  e.stopPropagation();
  if (t >= T - 1) t = 0;
  playing = !playing; playBtn.textContent = playing ? "Pause" : "Play";
};
document.getElementById("restart").onclick = () => {
  t = 0; playing = true; playBtn.textContent = "Pause";
};
document.getElementById("speed").oninput = e => fps = +e.target.value;
document.getElementById("show").onchange = e => limit = +e.target.value;
requestAnimationFrame(loop);
</script>
</body>
</html>
"""


def build_html(entries, path, episodes):
    """entries: list of dicts with name, bot, per_ep, hunter, prey, caught."""
    data = {
        "grid": GRID,
        "walls": [list(w) for w in sorted(WALLS)],
        "episodes": episodes,
        "agents": entries,
    }
    # json.dumps does not escape "</script>", which would close the script
    # block early and turn an agent name into live markup. "<" is inert
    # inside a JSON string and cannot break out of the tag.
    blob = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    html = TEMPLATE.replace("/*DATA*/", blob)
    with open(path, "w") as f:
        f.write(html)
    return path
