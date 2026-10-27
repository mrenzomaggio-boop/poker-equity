// Motore poker: parsing carte, valutazione mani, equity Monte Carlo.
const SUITS = "cdhs";
const RANKS = "  23456789TJQKA";

function parseCard(s) {
  const u = String(s).trim().toUpperCase();
  if (u.length < 2) return null;
  const suit = "CDHS".indexOf(u.slice(-1));
  if (suit < 0) return null;
  const rs = u.slice(0, -1);
  const map = { "10": 10, T: 10, J: 11, Q: 12, K: 13, A: 14 };
  let rank = map[rs];
  if (!rank) {
    const n = parseInt(rs, 10);
    if (n >= 2 && n <= 9 && String(n) === rs) rank = n; else return null;
  }
  return { rank, suit };
}

const cardCode = (c) => RANKS[c.rank] + SUITS[c.suit];

function popcount(x) { let n = 0; while (x) { x &= x - 1; n++; } return n; }

function straightHigh(mask) {
  let m = mask;
  if (m & (1 << 14)) m |= 1 << 1;
  for (let top = 14; top >= 5; top--) {
    const need = 0b11111 << (top - 4);
    if ((m & need) === need) return top;
  }
  return 0;
}

function pack(mask, n, excluding = []) {
  let out = 0, taken = 0;
  for (let r = 14; r >= 2 && taken < n; r--) {
    if ((mask & (1 << r)) && !excluding.includes(r)) { out = (out << 4) | r; taken++; }
  }
  return out;
}

// Punteggio della migliore mano da 5 su 5-7 carte (piu' alto = meglio).
function score(cards) {
  const counts = new Array(15).fill(0);
  const suitMask = [0, 0, 0, 0];
  let all = 0;
  for (const c of cards) { counts[c.rank]++; suitMask[c.suit] |= 1 << c.rank; all |= 1 << c.rank; }
  const quads = [], trips = [], pairs = [];
  for (let r = 14; r >= 2; r--) {
    if (counts[r] === 4) quads.push(r);
    else if (counts[r] === 3) trips.push(r);
    else if (counts[r] === 2) pairs.push(r);
  }
  const fs = [0, 1, 2, 3].find((s) => popcount(suitMask[s]) >= 5);
  if (fs !== undefined) { const sh = straightHigh(suitMask[fs]); if (sh) return (8 << 20) | sh; }
  if (quads.length) return (7 << 20) | (quads[0] << 4) | pack(all, 1, [quads[0]]);
  if (trips.length && (trips.length >= 2 || pairs.length)) {
    const p = Math.max(trips[1] || 0, pairs[0] || 0);
    return (6 << 20) | (trips[0] << 4) | p;
  }
  if (fs !== undefined) return (5 << 20) | pack(suitMask[fs], 5);
  const sh = straightHigh(all);
  if (sh) return (4 << 20) | sh;
  if (trips.length) return (3 << 20) | (trips[0] << 8) | pack(all, 2, [trips[0]]);
  if (pairs.length >= 2) return (2 << 20) | (pairs[0] << 8) | (pairs[1] << 4) | pack(all, 1, [pairs[0], pairs[1]]);
  if (pairs.length) return (1 << 20) | (pairs[0] << 12) | pack(all, 3, [pairs[0]]);
  return pack(all, 5);
}

const DECK = [];
for (let i = 0; i < 52; i++) DECK.push({ rank: 2 + (i >> 2), suit: i & 3 });

// Equity % (vittorie + meta' dei pareggi) contro `opponents` mani casuali.
function equity(hand, board, opponents, iterations = 2000) {
  const known = new Set([...hand, ...board].map((c) => c.rank * 4 + c.suit));
  const deck = DECK.filter((c) => !known.has(c.rank * 4 + c.suit));
  const need = 5 - board.length;
  const total = need + 2 * opponents;
  let wins = 0;
  for (let it = 0; it < iterations; it++) {
    for (let i = 0; i < total; i++) {           // shuffle parziale
      const j = i + Math.floor(Math.random() * (deck.length - i));
      [deck[i], deck[j]] = [deck[j], deck[i]];
    }
    const full = board.concat(deck.slice(0, need));
    const mine = score(hand.concat(full));
    let best = -1, tied = 0, idx = need;
    for (let o = 0; o < opponents; o++) {
      const sc = score([deck[idx], deck[idx + 1]].concat(full));
      idx += 2;
      if (sc > best) { best = sc; tied = 1; } else if (sc === best) tied++;
    }
    if (mine > best) wins += 1; else if (mine === best) wins += 1 / (tied + 1);
  }
  return (wins / iterations) * 100;
}

if (typeof module !== "undefined") module.exports = { parseCard, cardCode, score, equity };
