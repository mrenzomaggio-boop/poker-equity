// Riconoscimento carte "a logica": nessun modello e nessuna foto di addestramento.
// Cerca gli indici negli angoli delle carte (rango sopra seme, inchiostro scuro o colorato
// su fondo bianco) e li confronta con glifi generati al volo: font del dispositivo per i
// ranghi, font e sagome disegnate per i semi. La direzione rango -> seme dà l'inclinazione
// della carta, così anche le carte storte o capovolte vengono raddrizzate prima del confronto.
const Vision = (() => {
  const GW = 16, GH = 24, SS = 3;          // griglia di confronto e sottocampionamento
  const RANK_MIN = 0.55, SUIT_MIN = 0.55;  // somiglianza minima per accettare un indice
  const RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];
  const RANK_CODE = { "10": "T" };
  // Colore dell'inchiostro -> semi possibili (blu e verde: mazzo a quattro colori).
  const SUITS_BY_INK = { 2: "cs", 3: "dh", 4: "d", 5: "c" };
  // Mazzo a quattro colori impostato nel gioco: il colore da solo dice il seme.
  const FOUR_BY_INK = { 2: "s", 3: "h", 4: "d", 5: "c" };
  let four = false;
  const inkSuits = (c) => (four ? FOUR_BY_INK : SUITS_BY_INK)[c];
  const SUIT_CHARS = { s: "\u2660", c: "\u2663", h: "\u2665", d: "\u2666" };
  const FONTS = ["Arial", "Helvetica", "Verdana", "Trebuchet MS", "Georgia", "Times New Roman",
    "Courier New", "Impact", "Avenir Next", "Futura", "Gill Sans", "Chalkboard SE", "Marker Felt",
    "Noteworthy", "Comic Sans MS", "Roboto", "sans-serif", "serif"];
  const NB = [[-1, 0], [-2, 0], [-2, -1], [-1, -1], [0, -1], [1, -1], [2, -1], [-1, -2], [0, -2], [1, -2]];
  let rankT = null, suitT = null;
  const stats = { glyphs: 0, pairs: 0, sharp: 0 };

  // Campiona la forma su una griglia GWxGH nel sistema della carta (dn = verso il basso).
  // pts: pixel della forma [x0, y0, x1, y1, ...]; ink: intensità dell'inchiostro (larghezza w).
  // Si usano i livelli di grigio e non solo sì/no: con glifi di pochi pixel fa molta differenza.
  function feature(pts, ink, w, dx, dy) {
    const h = ink.length / w;
    const val = (x, y) => {
      const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
      if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) return 0;
      const i = y0 * w + x0;
      return (ink[i] * (1 - fx) + ink[i + 1] * fx) * (1 - fy) + (ink[i + w] * (1 - fx) + ink[i + w + 1] * fx) * fy;
    };
    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    for (let i = 0; i < pts.length; i += 2) {
      const u = pts[i] * dy - pts[i + 1] * dx, v = pts[i] * dx + pts[i + 1] * dy;
      if (u < u0) u0 = u; if (u > u1) u1 = u; if (v < v0) v0 = v; if (v > v1) v1 = v;
    }
    u0 -= 0.5; u1 += 0.5; v0 -= 0.5; v1 += 0.5;
    const su = (u1 - u0) / (GW * SS), sv = (v1 - v0) / (GH * SS);
    const f = new Float32Array(GW * GH);
    for (let j = 0; j < GH * SS; j++) {
      const v = v0 + (j + 0.5) * sv, row = ((j / SS) | 0) * GW;
      for (let i = 0; i < GW * SS; i++) {
        const u = u0 + (i + 0.5) * su;
        f[row + ((i / SS) | 0)] += val(u * dy + v * dx, v * dy - u * dx);
      }
    }
    // Leggera sfocatura della griglia: tollera tratti più sottili o più spessi dei modelli.
    const t = f.slice();
    for (let j = 0; j < GH; j++) for (let i = 0; i < GW; i++) {
      let a = 0, n = 0;
      for (let v = Math.max(0, j - 1); v <= Math.min(GH - 1, j + 1); v++) for (let u = Math.max(0, i - 1); u <= Math.min(GW - 1, i + 1); u++) { a += t[v * GW + u]; n++; }
      f[j * GW + i] = a / n;
    }
    let m = 0;
    for (let i = 0; i < f.length; i++) m += f[i];
    m /= f.length;
    let n = 0;
    for (let i = 0; i < f.length; i++) { f[i] -= m; n += f[i] * f[i]; }
    n = Math.sqrt(n) || 1;
    for (let i = 0; i < f.length; i++) f[i] /= n;
    return { f, aspect: (u1 - u0) / (v1 - v0) };
  }

  // Punteggio di ogni classe = media dei suoi K modelli più simili (un solo font "strano"
  // non basta a far vincere la classe sbagliata). I modelli imparati dal gioco stesso valgono
  // da soli: sono la stessa grafica, con la stessa sfocatura della fotocamera.
  function best(q, list, K = 3) {
    const by = {}, own = {};
    for (const t of list) {
      let s = 0;
      for (let i = 0; i < q.f.length; i++) s += q.f[i] * t.f[i];
      s -= 0.6 * Math.abs(Math.log(q.aspect / t.aspect));
      if (t.learned) { if (!(own[t.name] >= s)) own[t.name] = s; }
      else (by[t.name] = by[t.name] || []).push(s);
    }
    let top = { name: "", s: -9, q };
    for (const name of new Set([...Object.keys(by), ...Object.keys(own)])) {
      const v = (by[name] || []).sort((a, b) => b - a).slice(0, K);
      let s = v.length ? v.reduce((a, b) => a + b, 0) / v.length : -9;
      if (own[name] >= 0.75) s = Math.max(s, own[name] - 0.03);
      if (s > top.s) top = { name, s, q };
    }
    return top;
  }

  // Modelli imparati: numeri e semi del gioco reale, presi da letture confermate o corrette a mano.
  const learned = { rank: [], suit: [] }, KEEP = 4;
  function learn(kind, name, q) {
    if (!q) return;
    const list = learned[kind];
    list.push({ name, f: q.f, aspect: q.aspect, learned: true });
    const mine = list.filter((t) => t.name === name);
    if (mine.length > KEEP) list.splice(list.indexOf(mine[0]), 1);
  }
  // Una carta letta (rf/sf) diventa modello del codice indicato, es. "Td".
  function learnCard(code, rf, sf) {
    learn("rank", code[0] === "T" ? "10" : code[0], rf);
    learn("suit", code[1], sf);
  }
  const exportLearned = () => JSON.stringify(["rank", "suit"].map((k) =>
    learned[k].map((t) => [t.name, +t.aspect.toFixed(3), Array.from(t.f, (v) => +v.toFixed(3))])));
  function importLearned(json) {
    try {
      JSON.parse(json).forEach((list, i) => {
        learned[i ? "suit" : "rank"] = list.map(([name, aspect, f]) => ({ name, aspect, f: Float32Array.from(f), learned: true }));
      });
    } catch (e) {}
  }
  const forget = () => { learned.rank = []; learned.suit = []; };
  // Correzione da "wrong" a "right": i modelli imparati per il numero o il seme sbagliati vengono
  // probabilmente proprio da quella lettura sbagliata, quindi si tolgono.
  function unlearn(wrong, right) {
    const R = (c) => (c[0] === "T" ? "10" : c[0]);
    if (R(wrong) !== R(right)) learned.rank = learned.rank.filter((t) => t.name !== R(wrong));
    if (wrong[1] !== right[1]) learned.suit = learned.suit.filter((t) => t.name !== wrong[1]);
  }

  // Disegna con draw(g) su un canvas 96x96 e restituisce la forma come modello.
  function shape(name, draw) {
    const S = 96, c = document.createElement("canvas");
    c.width = c.height = S;
    const g = c.getContext("2d", { willReadFrequently: true });
    g.fillStyle = "#000"; g.textAlign = "center"; g.textBaseline = "middle";
    draw(g);
    const a = g.getImageData(0, 0, S, S).data, pts = [], ink = new Float32Array(S * S);
    for (let i = 0; i < S * S; i++) ink[i] = a[4 * i + 3];
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) if (ink[y * S + x] > 127) pts.push(x, y);
    if (pts.length < 40) return null;
    const t = feature(pts, ink, S, 0, 1);
    t.name = name;
    return t;
  }

  const circle = (g, x, y, r) => { g.moveTo(x + r, y); g.arc(x, y, r, 0, 2 * Math.PI); };
  const tri = (g, a, b, c) => { g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.lineTo(c[0], c[1]); g.closePath(); };
  const SUIT_PATHS = {
    d: (g) => { g.moveTo(48, 8); g.lineTo(76, 48); g.lineTo(48, 88); g.lineTo(20, 48); g.closePath(); },
    h: (g) => { circle(g, 34, 34, 16); circle(g, 62, 34, 16); tri(g, [19, 40], [77, 40], [48, 84]); },
    s: (g) => { circle(g, 34, 54, 15); circle(g, 62, 54, 15); tri(g, [20, 48], [76, 48], [48, 10]); tri(g, [48, 56], [38, 86], [58, 86]); },
    c: (g) => { circle(g, 48, 28, 15); circle(g, 31, 54, 15); circle(g, 65, 54, 15); tri(g, [48, 44], [38, 86], [58, 86]); },
  };

  function build() {
    rankT = []; suitT = [];
    const add = (list, t) => { if (t) list.push(t); };
    for (const font of FONTS) for (const w of ["300", "400", "700"]) {
      for (const r of RANKS) add(rankT, shape(r, (g) => { g.font = `${w} 64px "${font}"`; g.fillText(r, 48, 50); }));
      for (const s of "cdhs") add(suitT, shape(s, (g) => { g.font = `${w} 64px "${font}"`; g.fillText(SUIT_CHARS[s] + "\uFE0E", 48, 50); }));
    }
    for (const s of "cdhs") add(suitT, shape(s, (g) => { g.beginPath(); SUIT_PATHS[s](g); g.fill(); }));
  }

  // Sfocatura 3x3 di un canale (attenua il moiré dello schermo inquadrato).
  function blur(src, W, H) {
    const t = new Uint8Array(src.length), o = new Uint8Array(src.length);
    for (let y = 0; y < H; y++) {
      const r = y * W;
      t[r] = src[r]; t[r + W - 1] = src[r + W - 1];
      for (let x = 1; x < W - 1; x++) t[r + x] = (src[r + x - 1] + src[r + x] + src[r + x + 1]) / 3;
    }
    o.set(t.subarray(0, W)); o.set(t.subarray((H - 1) * W), (H - 1) * W);
    for (let i = W; i < (H - 1) * W; i++) o[i] = (t[i - W] + t[i] + t[i + W]) / 3;
    return o;
  }

  // opt.card: larghezza di una carta in pixel, quando si sa già dove sono le carte
  // (posizioni segnate): le misure dei glifi si regolano su quella invece che sull'immagine.
  function detect(img, opt = {}) {
    if (!rankT) build();
    const RT = rankT.concat(learned.rank), ST = suitT.concat(learned.suit);
    const W = img.width, H = img.height, N = W * H, d = img.data;
    let R = new Uint8Array(N), G = new Uint8Array(N), B = new Uint8Array(N);
    for (let i = 0; i < N; i++) { R[i] = d[4 * i]; G[i] = d[4 * i + 1]; B[i] = d[4 * i + 2]; }
    R = blur(R, W, H); G = blur(G, W, H); B = blur(B, W, H);

    // 1) Classi di pixel con soglie relative alla luce della scena:
    //    1 bianco (carta), 2 inchiostro scuro, 3 rosso, 4 blu, 5 verde.
    const lum = new Uint8Array(N), hist = new Uint32Array(256);
    for (let i = 0; i < N; i++) { const l = (R[i] * 77 + G[i] * 150 + B[i] * 29) >> 8; lum[i] = l; hist[l]++; }
    // Nitidezza del fotogramma (gradiente medio): l'app ignora i fotogrammi mossi.
    let gsum = 0;
    for (let y = 1; y < H; y += 2) for (let x = 1, i = y * W + 1; x < W; x += 2, i += 2) gsum += Math.abs(lum[i] - lum[i - 1]) + Math.abs(lum[i] - lum[i - W]);
    stats.sharp = gsum / (((H - 1) >> 1) * ((W - 1) >> 1) || 1);
    let L = 255;
    for (let l = 255, acc = 0; l >= 0; l--) { acc += hist[l]; if (acc > N * 0.02) { L = l; break; } }
    const tWhite = L * 0.7, tDark = L * 0.55;
    const ink = new Float32Array(N);
    for (let i = 0; i < N; i++) ink[i] = Math.max(0, L - lum[i]);
    // Soglia a isteresi: i pixel "deboli" (tratti sottili sbiaditi dalla sfocatura) tengono
    // insieme un glifo, ma un glifo deve contenere anche pixel decisi.
    const cls = new Uint8Array(N), strong = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      const r = R[i], g = G[i], b = B[i], mx = Math.max(r, g, b), sat = mx - Math.min(r, g, b);
      const tSat = Math.max(45, 0.35 * mx);
      if (lum[i] >= tWhite && sat < 0.35 * mx + 15) cls[i] = 1;
      else if (sat > 0.6 * tSat) { cls[i] = mx === r ? 3 : mx === b ? 4 : 5; strong[i] = sat > tSat; }
      else { cls[i] = 2; strong[i] = lum[i] < tDark; }
    }
    // Integrale dei pixel bianchi: dice in O(1) se attorno a un glifo c'è carta.
    const W1 = W + 1, wi = new Int32Array(W1 * (H + 1));
    for (let y = 0; y < H; y++) {
      let row = 0;
      for (let x = 0; x < W; x++) { row += cls[y * W + x] === 1; wi[(y + 1) * W1 + x + 1] = wi[y * W1 + x + 1] + row; }
    }
    const whites = (x0, y0, x1, y1) => {
      x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(W, x1); y1 = Math.min(H, y1);
      if (x1 <= x0 || y1 <= y0) return [0, 0];
      return [wi[y1 * W1 + x1] - wi[y0 * W1 + x1] - wi[y1 * W1 + x0] + wi[y0 * W1 + x0], (x1 - x0) * (y1 - y0)];
    };

    // 2) Componenti connesse dell'inchiostro (8-connesse, stesso colore).
    const lab = new Int32Array(N), parent = [0];
    const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, c = cls[i];
      if (c < 2) continue;
      let l = 0;
      // Vicini fino a 2 pixel: i tratti sottili spezzati da un pixel restano un unico glifo.
      for (const [ox, oy] of NB) {
        const xx = x + ox, yy = y + oy;
        if (xx < 0 || yy < 0 || xx >= W) continue;
        const j = yy * W + xx;
        if (cls[j] !== c) continue;
        const r = find(lab[j]);
        if (!l) l = r; else if (r !== l) { parent[r] = l; }
      }
      if (!l) { l = parent.length; parent.push(l); }
      lab[i] = l;
    }
    const comps = new Map();
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!lab[i]) continue;
      const r = find(lab[i]);
      lab[i] = r;
      let c = comps.get(r);
      if (!c) comps.set(r, c = { ids: [r], cls: cls[i], x0: x, x1: x, y0: y, y1: y, n: 0, ns: 0 });
      c.ns += strong[i];
      if (x < c.x0) c.x0 = x; if (x > c.x1) c.x1 = x; if (y < c.y0) c.y0 = y; if (y > c.y1) c.y1 = y;
      c.n++;
    }

    // 3) Glifi plausibili: dimensione, riempimento e soprattutto carta bianca tutt'intorno.
    const minS = opt.card ? Math.max(5, Math.round(0.08 * opt.card)) : Math.max(5, Math.round(Math.min(W, H) * 0.007));
    const maxS = opt.card ? 0.9 * opt.card : Math.min(W, H) * 0.15;
    // Con quattro colori il seme è già deciso dal colore: della forma basta che somigli a un seme.
    const suitMin = four ? 0.3 : SUIT_MIN, pairMin = four ? RANK_MIN + 0.3 : 2 * RANK_MIN + 0.03;
    const all = [...comps.values()];
    const onPaper = (c, s) => {
      const m = Math.ceil(0.3 * s);
      const [a, an] = whites(c.x0 - m, c.y0 - m, c.x1 + 1 + m, c.y1 + 1 + m);
      const [b, bn] = whites(c.x0, c.y0, c.x1 + 1, c.y1 + 1);
      return (a - b) / Math.max(1, an - bn) >= 0.45;
    };
    let glyphs = [];
    for (const c of all) {
      let w = c.x1 - c.x0 + 1, h = c.y1 - c.y0 + 1, s = Math.max(w, h);
      if (s < minS || s > maxS || c.absorbed || c.ns < 0.2 * c.n || c.n / (w * h) < 0.12 || !onPaper(c, s)) continue;
      // Pezzi staccati dalla soglia (il gambo sottile di picche e fiori, puntini) tornano al glifo.
      const gap = Math.max(2, Math.round(0.06 * s));
      for (const f of all) {
        if (f === c || f.absorbed || !f.ns || f.cls !== c.cls || f.n > 0.3 * c.n || Math.max(f.x1 - f.x0, f.y1 - f.y0) >= 0.6 * s) continue;
        if (Math.max(f.x0 - c.x1, c.x0 - f.x1, f.y0 - c.y1, c.y0 - f.y1) > gap) continue;
        f.absorbed = true;
        c.ids.push(...f.ids); c.n += f.n;
        c.x0 = Math.min(c.x0, f.x0); c.x1 = Math.max(c.x1, f.x1); c.y0 = Math.min(c.y0, f.y0); c.y1 = Math.max(c.y1, f.y1);
      }
      w = c.x1 - c.x0 + 1; h = c.y1 - c.y0 + 1; s = Math.max(w, h);
      const fill = c.n / (w * h);
      if (fill < 0.12 || fill > 0.92 || w / h < 0.12 || w / h > 2.5) continue;
      glyphs.push(c);
    }
    // Pezzi dello stesso carattere (tratti sottili spezzati) hanno riquadri che si sovrappongono: si uniscono.
    for (let i = 0; i < glyphs.length; i++) for (let j = 0; j < glyphs.length; j++) {
      const a = glyphs[i], b = glyphs[j];
      if (i === j || !a || !b || a.cls !== b.cls) continue;
      const ix = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0) + 1, iy = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0) + 1;
      if (ix <= 0 || iy <= 0) continue;
      const small = Math.min((a.x1 - a.x0 + 1) * (a.y1 - a.y0 + 1), (b.x1 - b.x0 + 1) * (b.y1 - b.y0 + 1));
      if (ix * iy < 0.3 * small) continue;
      // Solo pezzi di taglia simile: il seme piccolo dell'indice non va fuso col seme grande accanto.
      const sa = Math.max(a.x1 - a.x0, a.y1 - a.y0), sb = Math.max(b.x1 - b.x0, b.y1 - b.y0);
      if (Math.min(sa, sb) < 0.35 * Math.max(sa, sb)) continue;
      a.ids.push(...b.ids); a.n += b.n;
      a.x0 = Math.min(a.x0, b.x0); a.x1 = Math.max(a.x1, b.x1); a.y0 = Math.min(a.y0, b.y0); a.y1 = Math.max(a.y1, b.y1);
      glyphs[j] = null;
      j = -1;
    }
    glyphs = glyphs.filter(Boolean);
    // Il "10" è fatto di due pezzi affiancati: si uniscono se quello a sinistra è stretto come un "1".
    glyphs.sort((p, q) => p.x0 - q.x0);
    for (let i = 0; i < glyphs.length; i++) {
      const a = glyphs[i];
      if (!a || a.ten) continue;
      const ah = a.y1 - a.y0 + 1;
      const aw = a.x1 - a.x0 + 1;
      if (aw / ah > 0.7) continue;
      for (let j = 0; j < glyphs.length; j++) {
        const b = glyphs[j];
        if (!b || b === a || b.cls !== a.cls || b.ten) continue;
        const bh = b.y1 - b.y0 + 1, gap = b.x0 - a.x1;
        if (bh / ah < 0.75 || bh / ah > 1.33 || gap < -1 || gap > 0.5 * ah || b.x1 - b.x0 + 1 <= aw) continue;
        if (Math.abs((a.y0 + a.y1) - (b.y0 + b.y1)) / 2 > 0.3 * ah) continue;
        a.ids.push(...b.ids); a.n += b.n; a.ten = true;
        a.x0 = Math.min(a.x0, b.x0); a.x1 = Math.max(a.x1, b.x1); a.y0 = Math.min(a.y0, b.y0); a.y1 = Math.max(a.y1, b.y1);
        glyphs[j] = null;
        break;
      }
    }
    glyphs = glyphs.filter(Boolean);
    stats.glyphs = glyphs.length;
    for (const g of glyphs) {
      const set = g.ids;
      g.pts = [];
      for (let y = g.y0; y <= g.y1; y++) for (let x = g.x0; x <= g.x1; x++) if (set.includes(lab[y * W + x])) g.pts.push(x, y);
      // Senza i frammenti riattaccati: serve quando ha agganciato un'ombra o un pezzo di carta vicina.
      if (set.length > 1 && !g.ten) { g.core = []; for (let i = 0; i < g.pts.length; i += 2) if (lab[g.pts[i + 1] * W + g.pts[i]] === set[0]) g.core.push(g.pts[i], g.pts[i + 1]); }
      g.cx = (g.x0 + g.x1) / 2; g.cy = (g.y0 + g.y1) / 2; g.s = Math.max(g.x1 - g.x0, g.y1 - g.y0) + 1;
    }

    // 4) Coppie rango/seme dello stesso colore, una sopra l'altra (carta inclinata fino a 45°).
    //    Si provano entrambe le letture: rango sopra (angolo in alto) o seme sopra (carta capovolta).
    const cand = [];
    for (let i = 0; i < glyphs.length; i++) for (let j = i + 1; j < glyphs.length; j++) {
      const a = glyphs[i], b = glyphs[j];
      if (a.cls !== b.cls) continue;
      const smax = Math.max(a.s, b.s), ratio = a.s / b.s;
      let vx = b.cx - a.cx, vy = b.cy - a.cy;
      const dist = Math.hypot(vx, vy);
      if (ratio < 0.3 || ratio > 3.3 || dist < 0.5 * smax || dist > 2.2 * smax) continue;
      const [up, lo] = vy >= 0 ? [a, b] : [b, a];
      vx = (lo.cx - up.cx) / dist; vy = (lo.cy - up.cy) / dist;
      if (Math.abs(vx) > 0.71) continue;
      const suits = inkSuits(a.cls);
      for (const [rk, st, dx, dy] of [[up, lo, vx, vy], [lo, up, -vx, -vy]]) {
        const read = (g, list) => {
          const a = best(feature(g.pts, ink, W, dx, dy), list);
          if (!g.core || g.core.length < 0.5 * g.pts.length) return a;
          const b = best(feature(g.core, ink, W, dx, dy), list);
          return b.s > a.s ? b : a;
        };
        const r = read(rk, RT);
        const sT = ST.filter((t) => suits.includes(t.name));
        let s = read(st, sT);
        // Il seme piccolo dell'indice è di pochi pixel: se sulla stessa carta c'è un seme grande
        // (al centro, come nelle carte disegnate), quello è più leggibile e prevale.
        for (const g of glyphs) {
          if (g === rk || g === st || g.cls !== rk.cls || g.s < st.s * 1.4) continue;
          const ox = g.cx - rk.cx, oy = g.cy - rk.cy, u = ox * dy - oy * dx, v = ox * dx + oy * dy;
          if (v < 0.8 * rk.s || v > 6 * rk.s || u < -0.5 * rk.s || u > 5 * rk.s) continue;
          const big = best(feature(g.pts, ink, W, dx, dy), sT);
          if (big.s > s.s && big.s > 0.7) s = big;
        }
        // Un numero un po' incerto passa se il seme è chiaro (e viceversa): conta la coppia.
        if (r.s < RANK_MIN - 0.05 || s.s < suitMin || r.s + s.s < pairMin) continue;
        cand.push({ rk, st, code: (RANK_CODE[r.name] || r.name) + s.name, conf: (r.s + s.s) / 2, rf: r.q, sf: s.q });
      }
    }
    stats.pairs = cand.length;

    // 5) Ogni glifo appartiene a un solo indice; vince la lettura più sicura.
    cand.sort((p, q) => q.conf - p.conf);
    const used = new Set(), out = [];
    for (const c of cand) {
      if (used.has(c.rk) || used.has(c.st)) continue;
      used.add(c.rk); used.add(c.st);
      const x0 = Math.min(c.rk.x0, c.st.x0), x1 = Math.max(c.rk.x1, c.st.x1);
      const y0 = Math.min(c.rk.y0, c.st.y0), y1 = Math.max(c.rk.y1, c.st.y1);
      out.push({ code: c.code, conf: c.conf, rf: c.rf, sf: c.sf, x: (x0 + x1) / 2 / W, y: (y0 + y1) / 2 / H, w: (x1 - x0 + 1) / W, h: (y1 - y0 + 1) / H });
    }

    // 6) Numeri spezzati (righe del monitor, tratti sottili): si parte da un seme ben riconosciuto
    //    e si legge tutto l'inchiostro dello stesso colore subito sopra, anche se è in più pezzi.
    //    Vale per carte diritte, come quelle del board.
    for (const st of glyphs) {
      if (used.has(st)) continue;
      const s = best(feature(st.pts, ink, W, 0, 1), ST.filter((t) => inkSuits(st.cls).includes(t.name)));
      if (s.s < (four ? 0.4 : 0.6)) continue;
      const rx0 = st.cx - st.s, rx1 = st.cx + st.s, ry0 = st.y0 - 2.2 * st.s, ry1 = st.y0 - 1;
      const ids = new Set();
      let x0 = W, x1 = -1, y0 = H, y1 = -1, n = 0, taken = false;
      for (const c of all) {
        if (c.cls !== st.cls || c === st || c.x0 < rx0 || c.x1 > rx1 || c.y0 < ry0 || c.y1 > ry1) continue;
        if (used.has(c)) { taken = true; break; }  // sopra c'è già un indice letto: è il seme grande

        for (const id of c.ids) ids.add(id);
        n += c.n;
        x0 = Math.min(x0, c.x0); x1 = Math.max(x1, c.x1); y0 = Math.min(y0, c.y0); y1 = Math.max(y1, c.y1);
      }
      if (taken || n < 0.25 * st.n || y1 - y0 + 1 < 0.6 * st.s || y1 - y0 + 1 > 2 * st.s) continue;
      const pts = [];
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) if (ids.has(lab[y * W + x])) pts.push(x, y);
      const r = best(feature(pts, ink, W, 0, 1), RT);
      if (stats.debug) stats.debug.push(["anchor", Math.round(st.cx), Math.round(st.cy), r.name, r.s.toFixed(2), s.name, s.s.toFixed(2)]);
      if (r.s < RANK_MIN - 0.05) continue;
      used.add(st);
      const X0 = Math.min(x0, st.x0), X1 = Math.max(x1, st.x1);
      out.push({ code: (RANK_CODE[r.name] || r.name) + s.name, conf: (r.s + s.s) / 2, rf: r.q, sf: s.q, x: (X0 + X1) / 2 / W, y: (y0 + st.y1) / 2 / H, w: (X1 - X0 + 1) / W, h: (st.y1 - y0 + 1) / H });
    }
    return out;
  }

  const setFourColor = (v) => { four = !!v; };
  return { detect, stats, setFourColor, learnCard, unlearn, exportLearned, importLearned, forget, get learned() { return learned; } };
})();
