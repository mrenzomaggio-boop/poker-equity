// Riconoscimento carte (YOLO ONNX nel browser) + equity.
const SIZE = 640, CONF = 0.5, ITER = 2000, STABLE = 3;
const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay");
const work = document.createElement("canvas");
work.width = work.height = SIZE;
const wctx = work.getContext("2d", { willReadFrequently: true });

let session, labels, opponents = 1;
let hand = [], board = [], lastKey = "", streak = 0, published = "";
let lastBoxes = [];

const setStatus = (t) => { $("status").textContent = t; };

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
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
  } catch (e) { setStatus("Fotocamera non disponibile: " + e.message); $("start").disabled = false; return; }
  $("intro").hidden = true;
  $("intro").style.display = "none";
  setStatus("");
  loop();
}

function letterbox() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.min(SIZE / vw, SIZE / vh);
  const nw = Math.round(vw * s), nh = Math.round(vh * s);
  const px = (SIZE - nw) / 2, py = (SIZE - nh) / 2;
  wctx.fillStyle = "#727272";
  wctx.fillRect(0, 0, SIZE, SIZE);
  wctx.drawImage(video, px, py, nw, nh);
  return { px, py, nw, nh };
}

async function detect() {
  const lb = letterbox();
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

  const best = {}; // etichetta -> {conf, x, y, w, h} normalizzati sul frame
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
      x: (data[i] - lb.px) / lb.nw, y: (data[N + i] - lb.py) / lb.nh,
      w: data[2 * N + i] / lb.nw, h: data[3 * N + i] / lb.nh,
    };
  }
  return best;
}

function update(best) {
  const ranked = Object.entries(best).sort((a, b) => b[1].conf - a[1].conf);
  const h = ranked.filter(([, v]) => v.y >= 0.5).slice(0, 2).map(([k]) => k).sort();
  const b = ranked.filter(([, v]) => v.y < 0.5).slice(0, 5).map(([k]) => k).sort();
  lastBoxes = ranked.map(([k, v]) => ({ k, ...v }));
  const key = h.join("") + "|" + b.join("");
  if (key === lastKey) streak++; else { lastKey = key; streak = 1; }
  if (streak === STABLE && key !== published) { published = key; hand = h; board = b; recompute(); }
}

function recompute() {
  $("hand").textContent = hand.join(" ") || "-";
  $("board").textContent = board.join(" ") || "-";
  const hc = hand.map(parseCard), bc = board.map(parseCard);
  if (hc.length === 2 && (bc.length === 0 || bc.length >= 3)) {
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

$("pw").value = store.get("pe_pw") || "";
$("start").onclick = start;
$("minus").onclick = () => setOpp(opponents - 1);
$("plus").onclick = () => setOpp(opponents + 1);
