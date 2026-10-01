// Prints today's pregame rows as JSON, scored by the same engine the board uses.
// Usage: node monitor/snapshot.js data.json
const fs = require('fs'), path = require('path');
const { makeEngine } = require(path.join(__dirname, '..', 'engine.js'));
const DATA = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const E = makeEngine(DATA);
const r1 = v => v == null ? '' : Math.round(v * 10) / 10;
const rows = E.buildRows(DATA.today).map(r => ({
  date: DATA.today, gameId: r.g.id, state: r.g.state, start: r.g.start,
  team: r.ab, opp: r.opp, ha: r.ha,
  goalie_id: r.goalie ? r.goalie.id : '', goalie: r.goalie ? r.goalie.name : '', goalie_status: r.status,
  s_goalie: r1(r.s.goalie), s_def: r1(r.s.def), s_off: r1(r.s.off), s_st: r1(r.s.st), s_l5: r1(r.s.l5), s_fin: r1(r.s.fin),
  score: r1(r.total), proj: Math.round(r.lam * 1000) / 1000, p_over: Math.round((1 - r.u25) * 1000) / 1000,
  b2b: r.b2b ? 1 : 0, opp_b2b: r.oppB2b ? 1 : 0, model_version: DATA.model.version,
}));
process.stdout.write(JSON.stringify(rows));
