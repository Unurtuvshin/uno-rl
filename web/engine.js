// JavaScript port of game.py + env.py (state encoding) + opponents.py, plus greedy play of a saved Q table.
// Must stay identical to the Python: `python check_js.py` replays recorded Python games here and compares every step.
(function (root) {
'use strict';
const SKIP = 10, REV = 11, D2 = 12, WILD = 13, D4 = 14;
const KIND = [...Array(10).fill('NUM'), 'SKIP', 'REV', 'D2', 'WILD', 'D4'];
const DEFAULT_RULES = {
  n_players: 2, hand_size: 7, n_colors: 2, numbers: [1, 2], specials: ['SKIP', 'D2', 'D4'], hand_limit: 12,
  plus4_needs_no_color: false, stacking: [['D2', 'D2'], ['D4', 'D4'], ['D2', 'D4']],
  draw_even_if_playable: true, auto_play_drawn: false, max_turns: 500,
};
const FULL_RULES = { n_colors: 4, numbers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], specials: ['SKIP', 'REV', 'D2', 'WILD', 'D4'], hand_limit: 0 };
const key = c => c[0] * 16 + c[1];
const same = (a, b) => a[0] === b[0] && a[1] === b[1];

class Rng {                                   // mulberry32; only the page uses it (parity tests inject their own)
  constructor(seed) { this.s = (seed >>> 0) || 1; }
  random() { let t = this.s += 0x6D2B79F5; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }
  int(n) { return Math.floor(this.random() * n); }
  choice(a) { return a[this.int(a.length)]; }
  shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = this.int(i + 1); [a[i], a[j]] = [a[j], a[i]]; } }
}

function makeDeck(r) {
  const deck = [];
  const rep = (c, n) => { for (let i = 0; i < n; i++) deck.push([c[0], c[1]]); };
  for (let c = 0; c < r.n_colors; c++) {
    for (const v of r.numbers) rep([c, v], v === 0 ? 1 : 2);
    for (const s of r.specials) if (s === 'SKIP' || s === 'REV' || s === 'D2') rep([c, KIND.indexOf(s)], 2);
  }
  for (const s of r.specials) if (s === 'WILD' || s === 'D4') rep([4, KIND.indexOf(s)], 4);
  return deck;
}

class Game {
  constructor(rules, rng, start) {
    this.rules = rules; this.rng = rng;
    this.deck = makeDeck(rules); this.n_cards = this.deck.length;
    rng.shuffle(this.deck);
    const n = rules.n_players;
    const i0 = this.deck.findIndex(x => x[1] < 10);
    const c = this.deck.splice(i0, 1)[0];
    this.hands = [];
    for (let p = 0; p < n; p++) { const h = []; for (let k = 0; k < rules.hand_size; k++) h.push(this.deck.pop()); this.hands.push(h); }
    this.discard = [c]; this.top_color = c[0]; this.top_value = c[1];
    this.dir = 1; this.cur = (start || 0) % n;
    this.pend = 0; this.pend_kind = null; this.drawn = null;
    this.turns = 0; this.done = false; this.winner = null; this.observer = null;
  }
  snapshot(rng) {                              // everything needed to undo back to this moment
    return { deck: this.deck.map(c => [...c]), discard: this.discard.map(c => [...c]), hands: this.hands.map(h => h.map(c => [...c])),
      top_color: this.top_color, top_value: this.top_value, dir: this.dir, cur: this.cur, pend: this.pend, pend_kind: this.pend_kind,
      drawn: this.drawn ? [...this.drawn] : null, turns: this.turns, done: this.done, winner: this.winner, rngS: rng ? rng.s : null };
  }
  restore(sn, rng) {
    this.deck = sn.deck.map(c => [...c]); this.discard = sn.discard.map(c => [...c]); this.hands = sn.hands.map(h => h.map(c => [...c]));
    for (const k of ['top_color', 'top_value', 'dir', 'cur', 'pend', 'pend_kind', 'turns', 'done', 'winner']) this[k] = sn[k];
    this.drawn = sn.drawn ? [...sn.drawn] : null;
    if (rng && sn.rngS !== null) rng.s = sn.rngS;
  }
  total_cards() { return this.deck.length + this.discard.length + this.hands.reduce((s, h) => s + h.length, 0); }
  _advance(steps) { this.cur = (((this.cur + this.dir * (steps || 1)) % this.rules.n_players) + this.rules.n_players) % this.rules.n_players; this.drawn = null; }
  _draw(p, k) {
    const got = [];
    for (let i = 0; i < k; i++) {
      if (!this.deck.length && this.discard.length > 1) {
        const top = this.discard.pop();
        this.deck = this.discard; this.discard = [top];
        this.rng.shuffle(this.deck);
      }
      if (!this.deck.length) break;
      const card = this.deck.pop();
      this.hands[p].push(card); got.push(card);
    }
    const lim = this.rules.hand_limit;
    if (lim && this.hands[p].length >= lim) { this.done = true; this.winner = (p + 1) % this.rules.n_players; }
    return got;
  }
  _playable(card, hand) {
    const col = card[0], val = card[1];
    if (val === WILD) return true;
    if (val === D4) return !(this.rules.plus4_needs_no_color && hand.some(c => c[0] === this.top_color));
    return col === this.top_color || val === this.top_value;
  }
  _plays(card) {
    if (card[0] === 4) return Array.from({ length: this.rules.n_colors }, (_, c) => ['play', card, c]);
    return [['play', card, card[0]]];
  }
  best_color(p) {
    const counts = [0, 0, 0, 0];
    for (const c of this.hands[p]) if (c[0] < 4) counts[c[0]]++;
    let best = 0;
    for (let c = 1; c < this.rules.n_colors; c++) if (counts[c] > counts[best]) best = c;     // ties: lowest colour
    return best;
  }
  _unique(hand) { const seen = new Set(), out = []; for (const c of hand) if (!seen.has(key(c))) { seen.add(key(c)); out.push(c); } return out; }
  legal_moves(p) {
    const hand = this.hands[p], R = this.rules;
    if (this.pend) {
      const moves = [];
      for (const c of this._unique(hand))
        if (R.stacking.some(s => s[0] === this.pend_kind && s[1] === KIND[c[1]])) moves.push(...this._plays(c));
      moves.push(['draw']); return moves;
    }
    if (this.drawn !== null) return [...this._plays(this.drawn), ['pass']];
    const moves = [];
    for (const c of this._unique(hand)) if (this._playable(c, hand)) moves.push(...this._plays(c));
    if (!moves.length || R.draw_even_if_playable) moves.push(['draw']);
    return moves;
  }
  apply(mv) { const p = this.cur; this._apply(mv); if (this.observer) this.observer(this, p, mv); }
  _apply(mv) {
    const p = this.cur, kind = mv[0], R = this.rules;
    this.turns++;
    if (kind === 'play') {
      const card = mv[1], color = mv[2];
      const i = this.hands[p].findIndex(c => same(c, card)); this.hands[p].splice(i, 1);
      this.discard.push(card); this.top_color = color; this.top_value = card[1];
      if (!this.hands[p].length) { this.done = true; this.winner = p; return; }
      const v = card[1]; let step = 1;
      if (v === SKIP || (v === REV && R.n_players === 2)) step = 2;
      else if (v === REV) this.dir *= -1;
      else if (v === D2 || v === D4) { this.pend_kind = KIND[v]; this.pend += v === D2 ? 2 : 4; }
      this._advance(step);
    } else if (kind === 'draw') {
      if (this.pend) {
        this._draw(p, this.pend); this.pend = 0; this.pend_kind = null;
        if (!this.done) this._advance();
      } else {
        const got = this._draw(p, 1);
        if (this.done) return;
        if (got.length && this._playable(got[0], this.hands[p])) {
          if (R.auto_play_drawn) { this._apply(['play', got[0], got[0][0] < 4 ? got[0][0] : this.best_color(p)]); this.turns--; }
          else this.drawn = got[0];
        } else this._advance();
      }
    } else this._advance();
    if (!this.done && this.turns >= R.max_turns) this.done = true;
  }
}

// ---- opponents: f(game, seat, moves, rng) -> move
function randomOpp(g, p, moves, rng) { return rng.choice(moves); }
function ruleOpp(g, p, moves, rng) {
  const plays = moves.filter(m => m[0] === 'play');
  if (!plays.length) return moves[0];
  const small = Math.min(...g.hands.filter((_, i) => i !== p).map(h => h.length)) <= 2;
  const best = g.best_color(p);
  let bm = null, bs = -Infinity;
  for (const m of plays) {                    // same order and same one rng.random() per play as Python's max(key=...)
    const k = KIND[m[1][1]];
    let s = { NUM: 5, SKIP: 3, REV: 3, D2: 3, WILD: 1, D4: 0 }[k];
    if (small && (k === 'SKIP' || k === 'REV' || k === 'D2' || k === 'D4')) s += 10;
    if (g.pend) s += 20 + (k === 'D4' ? 1 : 0);
    s += 0.5 * (m[2] === best ? 1 : 0) + rng.random() * 0.1;
    if (s > bs) { bs = s; bm = m; }
  }
  return bm;
}
const OPPONENTS = { random: randomOpp, rule: ruleOpp };

// ---- env.py: abstraction of the game for seat 0
class Abstraction {
  constructor(r) {
    this.kinds = ['NUM', ...['SKIP', 'REV', 'D2', 'WILD', 'D4'].filter(k => r.specials.includes(k))];
    this.two = r.n_players === 2;
    if (this.two && this.kinds.includes('REV')) {
      this.kinds.splice(this.kinds.indexOf('REV'), 1);
      if (!this.kinds.includes('SKIP')) this.kinds.splice(1, 0, 'SKIP');
    }
    this.nc = r.n_colors; this.lim = r.hand_limit;
    this.DRAW = this.kinds.length * this.nc; this.PASS = this.DRAW + 1; this.nActions = this.PASS + 1;
    this.ranges = [this.lim ? this.lim - 1 : 5, this.lim ? this.lim - 1 : 4, this.nc, 4, 2, 2 ** this.kinds.length];
  }
  encode(g, seat) {                           // the state as seen by `seat` (default 0 = the agent); the game is symmetric
    seat = seat || 0;
    const hand = g.hands[seat], nc = this.nc;
    const counts = [0, 0, 0, 0];
    for (const c of hand) if (c[0] < 4) counts[c[0]]++;
    const order = [...Array(nc).keys()].sort((a, b) => (counts[b] - counts[a]) || (a - b));
    const rank = {}; order.forEach((c, i) => { rank[c] = i; });
    const by = new Map(); let bits = 0;
    for (const m of g.legal_moves(seat)) {
      let a;
      if (m[0] === 'draw') a = this.DRAW; else if (m[0] === 'pass') a = this.PASS;
      else {
        let k = KIND[m[1][1]];
        if (this.two && k === 'REV') k = 'SKIP';
        k = this.kinds.indexOf(k);
        a = k * nc + rank[m[2]]; bits |= 1 << k;
      }
      if (!by.has(a)) by.set(a, []);
      by.get(a).push(m);
    }
    const n = hand.length, o = Math.min(...g.hands.filter((_, i) => i !== seat).map(h => h.length));
    const pend = !g.pend ? 0 : g.pend === 2 ? 1 : g.pend === 4 ? 2 : 3;
    let nb, ob;
    if (this.lim) { nb = n - 1; ob = Math.min(o, this.lim - 1) - 1; }
    else { nb = n === 1 ? 0 : n === 2 ? 1 : n === 3 ? 2 : n <= 5 ? 3 : 4; ob = o === 1 ? 0 : o === 2 ? 1 : o <= 4 ? 2 : 3; }
    return { state: [nb, ob, rank[g.top_color], pend, g.drawn !== null ? 1 : 0, bits], by };
  }
  actionName(a) {
    if (a === this.DRAW) return 'draw';
    if (a === this.PASS) return 'pass';
    return this.kinds[Math.floor(a / this.nc)] + ' (colour#' + (a % this.nc + 1) + ')';
  }
}

// ---- a saved model (see train_all.py): sparse int16 Q table, Q*10000 (bucket brigade: the bid)
function b64bytes(s) { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
class Model {
  constructor(m) {
    this.meta = m; this.nA = m.n_actions; this.ranges = m.ranges;
    const idx = new Uint16Array(b64bytes(m.idx).buffer), val = new Int16Array(b64bytes(m.val).buffer);
    this.rows = new Map();
    for (let i = 0; i < idx.length; i++) this.rows.set(idx[i], val.subarray(i * this.nA, (i + 1) * this.nA));
  }
  q(state) {
    let i = 0; state.forEach((x, k) => { i = i * this.ranges[k] + x; });
    const r = this.rows.get(i), out = new Array(this.nA).fill(0);
    if (r) for (let a = 0; a < this.nA; a++) out[a] = r[a] / 10000;
    return out;
  }
  greedy(state, legal, rng) {                  // argmax over legal actions, random among ties (as Agent.select)
    const row = this.q(state); let best = -Infinity, c = [];
    for (const a of legal) { if (row[a] > best + 1e-12) { best = row[a]; c = [a]; } else if (Math.abs(row[a] - best) <= 1e-12) c.push(a); }
    return rng.choice(c);
  }
}

// ---- one game, recorded as events for the viewer: deal / decide / move / end
function recordGame(model, oppName, seed, rules, gameNo) {
  rules = rules || DEFAULT_RULES;
  const rng = new Rng(seed), ab = new Abstraction(rules), opp = OPPONENTS[oppName];
  let g = null, ev = [], hands = null, prevPend = 0;
  const snap = () => ({ hand: g.hands[0].map(c => [...c]), opp: g.hands.slice(1).map(h => h.length), top: [g.top_color, g.top_value], pend: g.pend, deck: g.deck.length });
  const counter = h => { const m = new Map(); for (const c of h) m.set(key(c), (m.get(key(c)) || 0) + 1); return m; };
  const observer = (gg, p, mv) => {
    const e = { t: 'move', p, k: mv[0], pend: 0, s: snap() };
    if (mv[0] === 'play') { e.card = [...mv[1]]; e.color = mv[2]; }
    else if (mv[0] === 'draw') {
      const before = counter(hands[p]), drawn = [];
      for (const c of gg.hands[p]) { const n = before.get(key(c)) || 0; if (n > 0) before.set(key(c), n - 1); else drawn.push([...c]); }
      e.drawn = drawn; e.n = drawn.length; e.pend = prevPend ? 1 : 0;
    }
    ev.push(e); hands = gg.hands.map(h => h.map(c => [...c])); prevPend = gg.pend;
  };
  const toAgent = () => { while (!g.done && g.cur !== 0) g.apply(opp(g, g.cur, g.legal_moves(g.cur), rng)); };
  for (let attempt = 0; ; attempt++) {         // a game the opponent wins before the agent ever moves is dealt again, as UnoEnv.reset does
    g = new Game(rules, rng, ((gameNo || 0) + attempt) % rules.n_players); ev = []; prevPend = 0; g.observer = observer;
    ev.push({ t: 'deal', s: snap() }); hands = g.hands.map(h => h.map(c => [...c]));
    toAgent();
    if (!g.done) break;
  }
  while (!g.done) {
    const { state, by } = ab.encode(g), legal = [...by.keys()];
    const row = model.q(state), act = model.greedy(state, legal, rng);
    const cq = {};
    for (const x of legal) for (const m of by.get(x)) if (m[0] === 'play') {
      const k = m[1][0] + ',' + m[1][1];
      if (!(k in cq) || row[x] > cq[k].q) cq[k] = { q: row[x], label: ab.actionName(x), col: m[1][0] === 4 ? m[2] : null, chosen: x === act };
    }
    ev.push({ t: 'decide', s: snap(), opts: legal.map(x => ({ label: ab.actionName(x), q: row[x], chosen: x === act })), cq,
      drawq: legal.includes(ab.DRAW) ? row[ab.DRAW] : null, passq: legal.includes(ab.PASS) ? row[ab.PASS] : null,
      drawchosen: act === ab.DRAW, passchosen: act === ab.PASS });
    g.apply(rng.choice(by.get(act)));
    toAgent();
  }
  const r = g.winner === null ? 0 : g.winner === 0 ? 1 : -1;
  const why = g.winner === null ? 'draw' : !g.hands[g.winner].length ? 'emptied' : 'limit';
  ev.push({ t: 'end', r, why, s: snap() });
  return ev;
}
// fast win rate of a model vs a bot (no events), for the page's own statistics
function simulate(model, oppName, n, seed, rules) {
  rules = rules || DEFAULT_RULES;
  const rng = new Rng(seed), ab = new Abstraction(rules), opp = OPPONENTS[oppName];
  let w = 0, l = 0;
  for (let i = 0, dealt = 0; i < n;) {
    const g = new Game(rules, rng, dealt++ % rules.n_players);
    const toAgent = () => { while (!g.done && g.cur !== 0) g.apply(opp(g, g.cur, g.legal_moves(g.cur), rng)); };
    toAgent();
    if (g.done) continue;                        // the opponent won before the agent moved: deal again (as UnoEnv.reset)
    while (!g.done) {
      const { state, by } = ab.encode(g), act = model.greedy(state, [...by.keys()], rng);
      g.apply(rng.choice(by.get(act))); toAgent();
    }
    if (g.winner === 0) w++; else if (g.winner !== null) l++;
    i++;
  }
  return { win: w / n, loss: l / n, draw: 1 - (w + l) / n };
}

const API = { Rng, Game, Abstraction, Model, DEFAULT_RULES, FULL_RULES, OPPONENTS, randomOpp, ruleOpp, makeDeck, recordGame, simulate, KIND, key };
if (typeof module !== 'undefined' && module.exports) module.exports = API; else root.UnoEngine = API;
})(typeof globalThis !== 'undefined' ? globalThis : this);
