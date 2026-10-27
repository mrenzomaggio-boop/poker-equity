// Riconoscimento carte (logica sugli indici, vision.js; YOLO ONNX solo con ?yolo) + equity.
const SIZE = 640, CONF = 0.35, ITER = 2000;
// Memoria della mano: ogni lettura nitida è un voto e una carta confermata (CONFIRM voti) resta
// finché non sparisce dall'inquadratura per FORGET_MS: basta leggerla bene qualche volta.
const CONFIRM = 3, FORGET_MS = 6000;
const useYolo = new URLSearchParams(location.search).has("yolo");
const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay");
const work = document.createElement("canvas");
work.width = work.height = SIZE;
const wctx = work.getContext("2d", { willReadFrequently: true });

let session, labels, opponents = 1;
let hand = [], board = [], memo = {}, alias = {}, sharpMax = 0, published = "";
// Correzione a mano: le carte scelte restano bloccate finché non si torna in automatico.
let lockHand = false, lockBoard = false, autoHand = [], autoBoard = [];
let lastBoxes = [];

const setStatus = (t) => { $("status").textContent = t; };

// Posizioni segnate una volta, con il telefono fermo su un supporto: le 2 carte della mano e le
// 3 del flop (coordinate del video 0..1). Turn e river stanno dopo il flop, allo stesso passo.
// Si legge solo lì, ingrandito, sommando gli ultimi AVG fotogrammi: le righe del monitor e il
// rumore cambiano da un fotogramma all'altro, le carte no.
const AVG = 3, TARGET = 360, ZONE_PX = 1.5e6;
let slots = null, marking = null;
const zbuf = { hand: [], board: [] }, zcan = { hand: document.createElement("canvas"), board: document.createElement("canvas") };
const MARK_STEPS = ["la tua prima carta", "la tua seconda carta", "la prima carta del flop (a sinistra)",
  "la seconda carta del flop", "la terza carta del flop"];

function zonesFrom(t, vw, vh) {
  const P = t.map((p) => ({ x: p.x * vw, y: p.y * vh }));
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const d = Math.max(8, (dist(P[2], P[3]) + dist(P[3], P[4])) / 2);
  const ux = (P[4].x - P[2].x) / (2 * d), uy = (P[4].y - P[2].y) / (2 * d);
  const bp = [P[2], P[3], P[4], { x: P[4].x + ux * d, y: P[4].y + uy * d }, { x: P[4].x + 2 * ux * d, y: P[4].y + 2 * uy * d }];
  const box = (pts, mx, my) => {
    const x0 = Math.max(0, Math.min(...pts.map((p) => p.x)) - mx), x1 = Math.min(vw, Math.max(...pts.map((p) => p.x)) + mx);
    const y0 = Math.max(0, Math.min(...pts.map((p) => p.y)) - my), y1 = Math.min(vh, Math.max(...pts.map((p) => p.y)) + my);
    return { x: Math.round(x0), y: Math.round(y0), w: Math.max(1, Math.round(x1 - x0)), h: Math.max(1, Math.round(y1 - y0)) };
  };
  return { d, hand: box([P[0], P[1]], 0.9 * d, 1.1 * d), board: box(bp, 0.75 * d, 1.0 * d) };
}

// Inquadratura cambiata (telefono spostato, carte nuove): la media ripartirebbe da immagini diverse.
function moved(a, b) {
  let s = 0, n = 0;
  for (let i = 0; i < a.length; i += 64) { s += Math.abs(a[i] - b[i]); n++; }
  return s / n > 12;
}

function detectZones() {
  const vw = video.videoWidth, vh = video.videoHeight, Z = zonesFrom(slots, vw, vh);
  const best = {}, t0 = performance.now();
  let glyphs = 0;
  for (const name of ["hand", "board"]) {
    const z = Z[name], cv = zcan[name];
    const k = Math.min(3, Math.max(1, TARGET / Z.d), Math.sqrt(ZONE_PX / (z.w * z.h)));
    const w = Math.round(z.w * k), h = Math.round(z.h * k);
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const g = cv.getContext("2d", { willReadFrequently: true });
    g.imageSmoothingQuality = "high";
    g.drawImage(video, z.x, z.y, z.w, z.h, 0, 0, w, h);
    const img = g.getImageData(0, 0, w, h), px = img.data, buf = zbuf[name];
    if (buf.length && (buf[0].length !== px.length || moved(buf[buf.length - 1], px))) buf.length = 0;
    buf.push(px.slice());
    if (buf.length > AVG) buf.shift();
    if (buf.length > 1) {
      const n = buf.length;
      for (let i = 0; i < px.length; i++) { let a = 0; for (let j = 0; j < n; j++) a += buf[j][i]; px[i] = a / n; }
    }
    for (const c of Vision.detect(img, { card: 0.95 * Z.d * k })) {
      c.x = (z.x + c.x * z.w) / vw; c.y = (z.y + c.y * z.h) / vh; c.w = c.w * z.w / vw; c.h = c.h * z.h / vh; c.zone = name;
      if (!best[c.code] || best[c.code].conf < c.conf) best[c.code] = c;
    }
    glyphs += Vision.stats.glyphs;
  }
  $("dbg").textContent = "simboli " + glyphs + " · carte " + Object.keys(best).length + " · " + Math.round(performance.now() - t0) + " ms";
  return best;
}

// Decifra cards.enc (salt 16 | iv 12 | AES-GCM) con la password.
async function decryptModel(pw) {
  const res = await fetch("cards.enc");
  if (!res.ok) throw new Error("modello non trovato");
  const buf = new Uint8Array(await res.arrayBuffer());
  const salt = buf.slice(0, 16), iv = buf.slice(16, 28), data = buf.slice(28);
  const km = await crypto.subtle.importKey("raw", new TextEncoder().encode(pw), "PBKDF2", false, ["deriveKey"]);
  const key = await crypto.subtle.deriveKey(
    { name: "PBKDF2", salt, iterations: 600000, hash: "SHA-256" },
    km, { name: "AES-GCM", length: 256 }, false, ["decrypt"]);
  try {
    return new Uint8Array(await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, data));
  } catch (e) { throw new Error("WRONG_PASSWORD"); }
}

const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} },
  del: (k) => { try { localStorage.removeItem(k); } catch (e) {} },
};

async function start() {
  const pw = $("pw").value;
  if (!pw) { setStatus("Inserisci la password"); return; }
  $("start").disabled = true;
  try {
    setStatus("Carico il modello...");
    labels = await (await fetch("labels.json")).json();
    const model = await decryptModel(pw);
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
  } catch (e) {
    if (e.message === "WRONG_PASSWORD") { store.del("pe_pw"); setStatus("Password errata"); }
    else setStatus("Errore modello: " + e.message);
    $("start").disabled = false;
    return;
  }
  if ($("remember").checked) store.set("pe_pw", pw); else store.del("pe_pw");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    // Schermo da inquadrare: meglio la messa a fuoco continua, dove il telefono la supporta.
    try { await stream.getVideoTracks()[0].applyConstraints({ advanced: [{ focusMode: "continuous" }] }); } catch (e) {}
  } catch (e) { setStatus("Cattura non disponibile: " + e.message); $("start").disabled = false; return; }
  $("intro").hidden = true;
  $("intro").style.display = "none";
  // iOS lascia la pagina scrollata dopo la tastiera della password: i pannelli finivano fuori schermo.
  if (document.activeElement) document.activeElement.blur();
  window.scrollTo(0, 0);
  setStatus("");
  loop();
}

// Disegna la regione r = {sx, sy, sw, sh} (pixel del video) nel canvas 640x640 con bande grigie.
function letterbox(r) {
  const s = Math.min(SIZE / r.sw, SIZE / r.sh);
  const nw = Math.round(r.sw * s), nh = Math.round(r.sh * s);
  const px = (SIZE - nw) / 2, py = (SIZE - nh) / 2;
  wctx.fillStyle = "#727272";
  wctx.fillRect(0, 0, SIZE, SIZE);
  wctx.drawImage(video, r.sx, r.sy, r.sw, r.sh, px, py, nw, nh);
  return { px, py, nw, nh, ...r };
}

// Inquadrando uno schermo le carte sono piccole: oltre al frame intero si analizzano
// due finestre quadrate sovrapposte lungo il lato lungo (sinistra/destra o alto/basso).
function regions() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const full = { sx: 0, sy: 0, sw: vw, sh: vh };
  if (vw > vh * 1.2) return [full, { sx: 0, sy: 0, sw: vh, sh: vh }, { sx: vw - vh, sy: 0, sw: vh, sh: vh }];
  if (vh > vw * 1.2) return [full, { sx: 0, sy: 0, sw: vw, sh: vw }, { sx: 0, sy: vh - vw, sw: vw, sh: vw }];
  return [full];
}

const frame = document.createElement("canvas");
const fctx = frame.getContext("2d", { willReadFrequently: true });

let topConf = 0;
async function detect() {
  if (slots) return detectZones();
  const vw = video.videoWidth, vh = video.videoHeight, k = Math.min(1, 1920 / Math.max(vw, vh));
  const fw = Math.round(vw * k), fh = Math.round(vh * k);
  if (frame.width !== fw || frame.height !== fh) { frame.width = fw; frame.height = fh; }
  fctx.drawImage(video, 0, 0, fw, fh);
  const t0 = performance.now(), best = {};
  for (const c of Vision.detect(fctx.getImageData(0, 0, fw, fh))) {
    if (!best[c.code] || best[c.code].conf < c.conf) best[c.code] = c;
  }
  const ms = Math.round(performance.now() - t0);
  topConf = 0;
  if (useYolo) for (const r of regions()) await detectRegion(r, best);
  $("dbg").textContent = "simboli " + Vision.stats.glyphs + " · carte " + Object.keys(best).length + " · " + ms + " ms" +
    (useYolo ? " · yolo " + topConf.toFixed(2) : "");
  return best;
}

async function detectRegion(region, best) {
  const lb = letterbox(region);
  const vw = video.videoWidth, vh = video.videoHeight;
  const d = wctx.getImageData(0, 0, SIZE, SIZE).data;
  const n = SIZE * SIZE;
  const f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    f[i] = d[4 * i] / 255;
    f[n + i] = d[4 * i + 1] / 255;
    f[2 * n + i] = d[4 * i + 2] / 255;
  }
  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", f, [1, 3, SIZE, SIZE]) });
  const o = out[session.outputNames[0]];
  const nc = o.dims[1] - 4, N = o.dims[2], data = o.data;
  // Diagnostica: miglior punteggio grezzo visto nel frame (aiuta a capire se il modello "vede" qualcosa).
  for (let i = 0; i < N; i++) for (let k = 0; k < nc; k++) { const v = data[(4 + k) * N + i]; if (v > topConf) topConf = v; }

  // best: etichetta -> {conf, x, y, w, h} normalizzati sul frame intero
  for (let i = 0; i < N; i++) {
    let c = 0, m = 0;
    for (let k = 0; k < nc; k++) { const v = data[(4 + k) * N + i]; if (v > m) { m = v; c = k; } }
    if (m < CONF) continue;
    const card = parseCard(labels[c]);
    if (!card) continue;
    const code = cardCode(card);
    if (best[code] && best[code].conf >= m) continue;
    best[code] = {
      conf: m,
      x: (lb.sx + (data[i] - lb.px) / lb.nw * lb.sw) / vw,
      y: (lb.sy + (data[N + i] - lb.py) / lb.nh * lb.sh) / vh,
      w: data[2 * N + i] / lb.nw * lb.sw / vw,
      h: data[3 * N + i] / lb.nh * lb.sh / vh,
    };
  }
}

const saveLearned = () => store.set("pe_learn", Vision.exportLearned());
const near = (a, b) => Math.abs(a.x - b.x) < 0.04 && Math.abs(a.y - b.y) < 0.03;
const rival = (k, m) => Object.entries(memo).some(([o, r]) => o !== k && r.n >= 2 && near(r, m));

function update(best) {
  const now = performance.now();
  // Fotogrammi mossi o sfocati (nitidezza molto sotto le migliori recenti) non votano,
  // salvo letture molto sicure: è lì che nascono gli scambi tipo 2 -> 7.
  // Con le posizioni segnate il telefono è fermo e i fotogrammi vengono mediati: votano tutti.
  sharpMax = Math.max(Vision.stats.sharp, sharpMax * 0.98);
  const sharp = !!slots || Vision.stats.sharp >= 0.7 * sharpMax;
  if (!sharp) $("dbg").textContent += " · tieni fermo il telefono";
  for (let [k, v] of Object.entries(best)) {
    if (!sharp && v.conf < 0.75) continue;
    if (alias[k] && memo[alias[k]] && near(v, memo[alias[k]])) k = alias[k];  // lettura già corretta a mano
    const m = memo[k] || (memo[k] = { n: 0, x: v.x, y: v.y });
    m.n++; m.seen = now;
    m.x += (v.x - m.x) * 0.3; m.y += (v.y - m.y) * 0.3;
    m.rf = v.rf; m.sf = v.sf; m.zone = v.zone;
    // Carta confermata più volte e senza letture rivali nello stesso punto: il suo numero e il suo
    // seme, così come li vede la fotocamera, diventano modelli per le prossime letture.
    if (sharp && v.conf >= 0.65 && (m.n === 6 || m.n === 20) && !rival(k, m)) { Vision.learnCard(k, v.rf, v.sf); saveLearned(); }
  }
  for (const k in memo) if (now - memo[k].seen > FORGET_MS) delete memo[k];
  for (const k in alias) if (!memo[alias[k]]) delete alias[k];
  // Letture diverse nello stesso punto (es. 2♦ e 7♦): vince quella vista più volte.
  const kept = [];
  for (const [k, m] of Object.entries(memo).filter(([, m]) => m.n >= CONFIRM).sort((a, b) => b[1].n - a[1].n)) {
    if (!kept.some(([, o]) => near(o, m))) kept.push([k, m]);
  }
  const inHand = (m) => (slots ? m.zone === "hand" : m.y >= 0.5);
  const h = kept.filter(([, m]) => inHand(m)).slice(0, 2).map(([k]) => k).sort();
  const b = kept.filter(([, m]) => !inHand(m)).slice(0, 5).map(([k]) => k).sort();
  lastBoxes = Object.entries(best).map(([k, v]) => ({ k, ...v }));
  autoHand = h; autoBoard = b;
  const nh = lockHand ? hand : h, nb = lockBoard ? board : b;
  const key = nh.join("") + "|" + nb.join("");
  if (key !== published) { published = key; hand = nh; board = nb; recompute(); }
}

const SUIT_SYM = { s: "\u2660", h: "\u2665", d: "\u2666", c: "\u2663" };
const pretty = (k) => k[0] + SUIT_SYM[k[1]];

function recompute() {
  $("hand").textContent = (hand.map(pretty).join(" ") || "-") + (lockHand ? " \u{1F512}" : "");
  $("board").textContent = (board.map(pretty).join(" ") || "-") + (lockBoard ? " \u{1F512}" : "");
  const hc = hand.map(parseCard), bc = board.map(parseCard);
  // La probabilità serve appena si conoscono le due carte in mano, qualunque sia il board.
  if (hc.length === 2) {
    $("eq").textContent = equity(hc, bc, opponents, ITER).toFixed(1) + "%";
  } else {
    $("eq").textContent = "";
  }
}

function draw() {
  const dw = overlay.clientWidth, dh = overlay.clientHeight;
  overlay.width = dw; overlay.height = dh;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, dw, dh);
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.max(dw / vw, dh / vh);
  const ox = (dw - vw * s) / 2, oy = (dh - vh * s) / 2;
  if (slots) {
    const Z = zonesFrom(slots, vw, vh);
    ctx.strokeStyle = "rgba(255, 235, 59, .8)"; ctx.lineWidth = 2;
    for (const z of [Z.hand, Z.board]) ctx.strokeRect(ox + z.x * s, oy + z.y * s, z.w * s, z.h * s);
  }
  ctx.fillStyle = "#ff0";
  for (const p of marking || []) { ctx.beginPath(); ctx.arc(ox + p.x * vw * s, oy + p.y * vh * s, 8, 0, 2 * Math.PI); ctx.fill(); }
  ctx.strokeStyle = "#3f3"; ctx.fillStyle = "#3f3"; ctx.lineWidth = 2; ctx.font = "14px monospace";
  for (const b of lastBoxes) {
    const x = ox + (b.x - b.w / 2) * vw * s, y = oy + (b.y - b.h / 2) * vh * s;
    ctx.strokeRect(x, y, b.w * vw * s, b.h * vh * s);
    ctx.fillText(b.k, x, y - 4);
  }
}

let busy = false, lastRun = 0;
async function loop(ts = 0) {
  if (!busy && ts - lastRun > 150 && video.videoWidth) {
    busy = true; lastRun = ts;
    try { update(await detect()); draw(); } catch (e) { setStatus(e.message); }
    busy = false;
  }
  requestAnimationFrame(loop);
}

function setOpp(n) {
  opponents = Math.min(8, Math.max(1, n));
  $("opp").textContent = opponents;
  recompute();
}

// Selettore: 2 caselle per la mano e 5 per il board; si tocca una casella e poi la carta.
let draft = [], slot = 0;
function openPicker() {
  draft = [hand[0], hand[1], ...board];
  while (draft.length < 7) draft.push(null);
  slot = Math.max(0, draft.indexOf(null));
  renderPicker();
  $("picker").hidden = false;
}
function renderPicker() {
  document.querySelectorAll("#picker .slot").forEach((el) => {
    const i = +el.dataset.i;
    el.textContent = draft[i] ? pretty(draft[i]) : "\u00b7";
    el.className = "slot" + (i === slot ? " sel" : "") + (draft[i] && "hd".includes(draft[i][1]) ? " red" : "");
  });
  const grid = $("grid");
  grid.textContent = "";
  for (const s of "shdc") for (const r of "AKQJT98765432") {
    const k = r + s, b = document.createElement("button");
    b.textContent = pretty(k);
    b.className = "hd".includes(s) ? "red" : "";
    b.disabled = draft.includes(k) && draft[slot] !== k;
    b.onclick = () => {
      draft[slot] = k;
      const next = draft.indexOf(null, slot + 1);
      slot = next >= 0 ? next : slot;
      renderPicker();
    };
    grid.appendChild(b);
  }
}
document.querySelectorAll("#picker .slot").forEach((el) => {
  el.onclick = () => { const i = +el.dataset.i; if (i === slot) draft[i] = null; slot = i; renderPicker(); };
});
$("info").onclick = openPicker;
// Correzione a mano di una sola carta letta male: l'app impara come appare davvero quella carta
// e corregge la sua memoria, così non serve bloccare. Negli altri casi la sezione resta bloccata.
function teach(auto, fixed) {
  const rem = auto.filter((k) => !fixed.includes(k)), add = fixed.filter((k) => !auto.includes(k));
  if (rem.length !== 1 || add.length !== 1 || !memo[rem[0]]) return false;
  const m = memo[rem[0]];
  Vision.unlearn(rem[0], add[0]);
  if (m.rf) Vision.learnCard(add[0], m.rf, m.sf);
  saveLearned();
  memo[add[0]] = { ...m, n: m.n + 5 };
  delete memo[rem[0]];
  alias[rem[0]] = add[0];
  return true;
}
$("pick-ok").onclick = () => {
  const h = draft.slice(0, 2).filter(Boolean).sort(), b = draft.slice(2).filter(Boolean).sort();
  if (h.join() !== hand.slice().sort().join()) lockHand = !teach(autoHand, h);
  if (b.join() !== board.slice().sort().join()) lockBoard = !teach(autoBoard, b);
  hand = h; board = b; published = "";
  $("picker").hidden = true;
  recompute();
};
$("pick-auto").onclick = () => {
  lockHand = lockBoard = false;
  hand = autoHand; board = autoBoard; published = "";
  $("picker").hidden = true;
  recompute();
};
$("pick-close").onclick = () => { $("picker").hidden = true; };
$("pick-forget").onclick = () => { Vision.forget(); store.del("pe_learn"); $("picker").hidden = true; };
Vision.importLearned(store.get("pe_learn") || "[]");

// Segnare le posizioni: si tocca il centro di 5 carte sull'immagine della fotocamera.
const resetMemory = () => { memo = {}; alias = {}; zbuf.hand = []; zbuf.board = []; published = ""; };
function setSlots(v) {
  slots = v;
  if (v) store.set("pe_slots", JSON.stringify(v)); else store.del("pe_slots");
  $("divider").hidden = !!v;
  $("mark").textContent = v ? "Rifai posizioni" : "Segna carte";
  resetMemory();
}
function markHint() {
  $("markhint").hidden = !marking;
  if (marking) $("marktext").textContent = "Tocca il centro della " + MARK_STEPS[marking.length].replace(/^(la|il) /, "");
}
$("mark").onclick = () => { marking = []; markHint(); };
$("mark-cancel").onclick = () => { marking = null; markHint(); };
$("mark-clear").onclick = () => { marking = null; markHint(); setSlots(null); };
overlay.addEventListener("click", (ev) => {
  if (!marking || !video.videoWidth) return;
  const r = overlay.getBoundingClientRect(), vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.max(r.width / vw, r.height / vh), ox = (r.width - vw * s) / 2, oy = (r.height - vh * s) / 2;
  marking.push({ x: (ev.clientX - r.left - ox) / s / vw, y: (ev.clientY - r.top - oy) / s / vh });
  if (marking.length === MARK_STEPS.length) { setSlots(marking); marking = null; }
  markHint();
  draw();
});
try { const v = JSON.parse(store.get("pe_slots")); if (Array.isArray(v) && v.length === 5) setSlots(v); } catch (e) {}

// Mazzo a quattro colori (opzione del gioco): il seme si legge dal colore.
function setFour(v) {
  Vision.setFourColor(v);
  $("four").checked = v;
  if (v) store.set("pe_four", "1"); else store.del("pe_four");
  resetMemory();
}
$("four").onchange = () => setFour($("four").checked);
setFour(store.get("pe_four") === "1");

window.addEventListener("scroll", () => { if (window.scrollY || window.scrollX) window.scrollTo(0, 0); });

$("pw").value = store.get("pe_pw") || "";
$("start").onclick = start;
$("minus").onclick = () => setOpp(opponents - 1);
$("plus").onclick = () => setOpp(opponents + 1);
