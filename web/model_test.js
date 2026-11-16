// node web/model_test.js : does the JS decode + greedy play of every saved model reproduce the win rates Python measured?
const fs = require('fs'), path = require('path'), E = require('./engine.js');
global.atob = s => Buffer.from(s, 'base64').toString('binary');
const dir = path.join(__dirname, '..', 'models');
let worst = 0, n = 0, bad = 0;
const N = +process.argv[2] || 3000;
for (const f of fs.readdirSync(dir).filter(x => x.endsWith('.json'))) {
  const m = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8')), model = new E.Model(m);
  for (const opp of ['random', 'rule']) {
    const r = E.simulate(model, opp, N, 12345), py = m.eval[opp].win;
    const d = Math.abs(r.win - py); worst = Math.max(worst, d); n++; sum = (global.sum || 0) + (r.win - py);
    if (d > 0.045) { bad++; console.log('DIFF', f, opp, 'js', r.win.toFixed(3), 'py', py.toFixed(3)); }
  }
}
console.log("mean signed difference (js - py):", (100 * sum / n).toFixed(2), "points"); console.log(`${n} model/opponent pairs, ${N} games each in JS vs Python's 2000: largest difference ${(100 * worst).toFixed(1)} points, ${bad} above 4.5 points`);
