// node web/parity_test.js parity.json   (run through check_js.py)
const fs = require('fs'), E = require('./engine.js');
const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const canon = m => JSON.stringify([m[0]].concat(m[0] === 'play' ? [m[1], m[2]] : []));
const sortedHands = g => g.hands.map(h => h.map(c => [...c]).sort((a, b) => a[0] - b[0] || a[1] - b[1]));
let steps = 0, fails = 0;
const fail = (msg) => { if (fails++ < 10) console.log('MISMATCH', msg); };
for (const [ci, c] of cases.entries()) {
  const rules = Object.assign({}, c.rules);
  let noise = [], shuf = c.game.shuffles.slice();
  const rng = { shuffle(a) { const n = shuf.shift(); if (!n || n.length !== a.length) throw new Error('shuffle queue'); a.splice(0, a.length, ...n.map(x => [...x])); },
    random() { return noise.shift(); }, choice() { throw new Error('choice not scripted'); } };
  const g = new E.Game(rules, rng, c.game.start), ab = new E.Abstraction(rules);
  const tag = `case ${ci} (${c.variant})`;
  if (g.n_cards !== c.game.initial_cards) fail(tag + ' deck size');
  for (const [si, st] of c.game.steps.entries()) {
    steps++;
    const legal = g.legal_moves(g.cur).map(canon).sort();
    if (g.cur !== st.seat) { fail(`${tag} step ${si}: seat ${g.cur} vs ${st.seat}`); break; }
    if (JSON.stringify(legal) !== JSON.stringify(st.legal)) { fail(`${tag} step ${si}: legal moves\n js ${legal}\n py ${st.legal}`); break; }
    {
      const { state, by } = ab.encode(g, st.seat);
      if (JSON.stringify(state) !== JSON.stringify(st.state)) { fail(`${tag} step ${si}: state ${state} vs ${st.state}`); break; }
      const acts = [...by.keys()].sort((a, b) => a - b);
      if (JSON.stringify(acts) !== JSON.stringify(st.actions)) { fail(`${tag} step ${si}: actions ${acts} vs ${st.actions}`); break; }
    }
    if (st.seat !== 0) {
      if (st.opp === 'rule') {                 // the random bot's choice is just an rng draw: nothing to compare
        noise = st.noise.slice();
        const ordered = st.order.map(x => { const m = JSON.parse(x); return m[0] === 'play' ? ['play', m[1], m[2]] : [m[0]]; });   // Python's move order
        const mv = E.OPPONENTS.rule(g, g.cur, ordered, rng);
        if (canon(mv) !== st.move) { fail(`${tag} step ${si}: rule bot chose ${canon(mv)} vs ${st.move}`); break; }
      }
    }
    const mv = JSON.parse(st.move);
    g.apply(mv[0] === 'play' ? ['play', mv[1], mv[2]] : [mv[0]]);
    const a = st.after;
    const mine = JSON.stringify([sortedHands(g), [g.top_color, g.top_value], g.pend, g.pend_kind, g.deck.length, g.discard.length, g.cur, g.drawn, g.done, g.winner]);
    const theirs = JSON.stringify([a.hands, a.top, a.pend, a.kind, a.deck, a.disc, a.cur, a.drawn, a.done, a.winner]);
    if (mine !== theirs) { fail(`${tag} step ${si}: state after move\n js ${mine}\n py ${theirs}`); break; }
    if (g.total_cards() !== g.n_cards) { fail(tag + ' card count'); break; }
  }
}
console.log(`${cases.length} games, ${steps} steps compared, ${fails} mismatches`);
process.exit(fails ? 1 : 0);
