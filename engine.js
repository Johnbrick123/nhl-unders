// The one scoring engine. The board inlines this file; monitor/snapshot.js runs it in Node
// to log each day's pregame rows, so the log always matches what the board showed.
// Weights and projection settings come from model.json (DATA.model).
function makeEngine(DATA){
  const M=DATA.model, P=M.projection, T=DATA.teams, AB=Object.keys(T), G=DATA.goalies, REC=DATA.recent||{};
  const mean=f=>AB.reduce((s,a)=>s+f(T[a]),0)/AB.length;
  const LG={gf:mean(t=>t.gf),xg:mean(t=>t.xgf),pp:mean(t=>t.ppPct),pk:mean(t=>t.pkPct),ppg:mean(t=>t.ppg)};
  for(const a of AB){const t=T[a];
    t.off=0.65*t.xgf/LG.xg+0.35*t.gf/LG.gf; t.def=t.xga/LG.xg;
    t.shPct=t.sf?t.gf/t.sf:0; t.fin=t.xgf?t.gf/t.xgf:1;
    const r=REC[a]||[]; t.l5=r.length>=3?r.reduce((s,x)=>s+(x[0]||0),0)/r.length:null; t.l5n=r.length;}
  for(const g of G){const rate=g.xga?g.gsax/g.xga:0,w=g.xga/(g.xga+60);
    g.mult=g.xga?Math.min(1.15,Math.max(0.85,1-w*rate)):1; g.gp=g.gpPrior+g.gpCur;}
  const byTeam={}; for(const g of G)(byTeam[g.team]=byTeam[g.team]||[]).push(g);
  for(const a in byTeam)byTeam[a].sort((x,y)=>y.gp-x.gp);

  // 0-100 scales, high = favors scoring
  const span=f=>{const v=AB.map(a=>f(T[a]));return{min:Math.min(...v),max:Math.max(...v)}};
  const gq=G.filter(g=>g.gp>=15).map(g=>g.mult);
  const SC={off:span(t=>t.off),def:span(t=>t.xga),st:{min:0.55,max:1.55},fin:span(t=>t.fin),l5:{min:1.5,max:4.5},
    goalie:gq.length?{min:Math.min(...gq),max:Math.max(...gq)}:{min:0.9,max:1.1}};
  const clamp=v=>Math.max(0,Math.min(100,v));
  const lowGood=(v,s)=>v==null?null:clamp((s.max-v)/(s.max-s.min)*100);
  const highGood=(v,s)=>v==null?null:clamp((v-s.min)/(s.max-s.min)*100);

  const norm=s=>(s||'').toLowerCase().replace(/[^a-z]/g,'');
  const nameToAb={}; for(const a of AB)nameToAb[norm(T[a].name)]=a;
  function projectedStarter(date,ab){
    const day=(DATA.starters||{})[date]||{};
    for(const k in day){if(nameToAb[norm(k)]===ab){
      const nm=norm(day[k].name),list=byTeam[ab]||[];
      const g=list.find(x=>norm(x.name)===nm)||list.find(x=>nm.endsWith(norm(x.last)));
      if(g)return{g,status:day[k].status};}}
    return{g:(byTeam[ab]||[])[0],status:'Default'};
  }
  const pois=(k,l)=>{let p=Math.exp(-l),s=p;for(let i=1;i<=k;i++){p*=l/i;s+=p}return s};
  const american=p=>p>=0.5?String(Math.round(-100*p/(1-p))):'+'+Math.round(100*(1-p)/p);
  const playedOn=(d,ab)=>(DATA.schedule[d]||[]).some(g=>g.away===ab||g.home===ab);
  const prevDay=d=>{const x=new Date(d+'T12:00:00Z');x.setUTCDate(x.getUTCDate()-1);return x.toISOString().slice(0,10)};

  function buildRows(date,W,overrides){
    W=W||M.weights; overrides=overrides||{};
    const pd=prevDay(date),known=pd in DATA.schedule,out=[];
    for(const g of DATA.schedule[date]||[])for(const side of ['away','home']){
      const ab=side==='away'?g.away:g.home, opp=side==='away'?g.home:g.away, t=T[ab], o=T[opp];
      if(!t||!o)continue;
      const key=`${date}|${g.id}|${opp}`, ps=projectedStarter(date,opp);
      const gid=overrides[key], goalie=(gid&&G.find(x=>x.id===gid))||ps.g;
      const status=gid&&ps.g&&gid!==ps.g.id?'Your pick':ps.status;
      const b2b=known&&playedOn(pd,ab), oppB2b=known&&playedOn(pd,opp);
      const st=(t.ppg/LG.ppg)*(o.ppga/LG.ppg);          // PP goals/gm vs opponent's PK goals allowed/gm
      const gm=goalie?goalie.mult:1;
      const raw=LG.gf*t.off*o.def*gm*(side==='home'?P.home:P.away)*(b2b?P.b2b:1)*(oppB2b?P.oppB2b:1);
      const lam=LG.gf+P.shrink*(raw-LG.gf);
      const s={goalie:highGood(gm,SC.goalie),def:highGood(o.xga,SC.def),off:highGood(t.off,SC.off),
               st:highGood(st,SC.st),l5:highGood(t.l5,SC.l5),fin:highGood(t.fin,SC.fin)};
      let num=0,den=0,partial=false; for(const k in W){if(s[k]==null){if(W[k]>0)partial=true;continue}num+=s[k]*W[k];den+=W[k]}
      const fin=['FINAL','OFF'].includes(g.state), actual=side==='away'?g.awayScore:g.homeScore;
      out.push({g,key,ab,opp,t,o,ha:side==='home'?'Home':'Away',game:`${g.away} @ ${g.home}`,goalie,status,b2b,oppB2b,st,lam,
        u25:pois(2,lam),u35:pois(3,lam),s,total:den?num/den:null,partial,final:fin,actual});
    }
    return out;
  }
  return {M,T,AB,G,REC,LG,SC,byTeam,lowGood,highGood,projectedStarter,pois,american,prevDay,buildRows};
}
if(typeof module!=='undefined')module.exports={makeEngine};
