// Session-local computations for the original lab pages.
window.showButtonProgress = function (button, label) {
  const nodes = [...button.childNodes];
  button.textContent = label;
  button.setAttribute('aria-busy', 'true');
  return () => {
    if (button.textContent === label) button.replaceChildren(...nodes);
    button.removeAttribute('aria-busy');
  };
};

window.bindModuleUndo = function (button, status, module, action) {
  let running = false;
  function sync() {
    const runtime = window.labRuntime;
    const target = runtime.undoStatus(module);
    button.hidden = !runtime.canWrite(action);
    status.hidden = button.hidden;
    button.disabled = running || !target;
    button.title = target ? '回滚：' + target.label : '本页暂无回滚记录';
  }
  button.addEventListener('click', async () => {
    const runtime = window.labRuntime;
    const target = runtime.undoStatus(module);
    if (running || !target || !button.ownerDocument.defaultView.confirm('回滚本页上一次写入（' + target.label + '）？')) return;
    running = true;
    sync();
    const finish = window.showButtonProgress(button, '回滚中…');
    status.textContent = '正在回滚…';
    try {
      const result = await runtime.undo({module, id:target.id});
      status.textContent = '已回滚：' + result.undone;
    } catch (error) {
      status.textContent = error.message;
    } finally {
      running = false;
      finish();
      sync();
    }
  });
  sync();
  return sync;
};

window.createLabRuntime = function (snapshot, payload, demo, readSourceModule, session, mutate) {
  const visible = snapshot.items.filter(item => item.k !== "私密");
  const byStrength = [...visible].sort((a, b) => b.s - a.s);
  const { history, cards, added } = session.cards ? session : Object.assign(session, {
    history: [[0, 0, 0], [0, 0, 0]], cards: [], added: new Set(), seq: 0, batches: 0
  });
  const details = payload.details || {};
  const write = payload.write || {};
  const canWrite = action => write.actions?.includes(action) === true;
  const writableRoutes = {
    "/api/settings": "settings", "/api/handover": "handover", "/api/dsh": "dsh",
    "/api/cards/generate": "generate", "/api/cards/review": "review",
    "/api/cards/maintain": "maintain", "/api/cards/outcome": "outcome"
  };
  const num = value => Number.isFinite(Number(value)) ? Number(value) : 0;
  const clip = (value, count) => Array.from(String(value ?? "").trim()).slice(0, count).join("");
  const chars = value => Array.from(value).length;
  const sum = series => series.reduce((a, b) => a + b, 0);
  const round = value => Number(value.toFixed(1));
  const DATE = d => new Date(Date.now() - d * 864e5).toISOString().slice(0,10);
  const samples = [
    ["active", r => r.p === 0 && !r.ep, "活跃记忆"],
    ["intimate", r => r.p === 2, "私密记忆（隐藏）"],
    ["episode", r => r.ep, "情景记忆"],
    ["short", r => r.p === 1, "短期记忆"],
    ["cold", r => r.p === 3, "冷存记忆"]
  ].map(([mode, match, fallback]) => {
    const item = snapshot.items.find(match);
    const s = item?.s ?? 0, im = item?.im || 0, em = item?.em || 0;
    const track = (current, spread) => Array.from({length:35}, (_, i) => {
      if (i === 34) return current;
      return demo && item && i >= 10
        ? Math.max(0, Math.min(100, round(current + spread * Math.sin(i / 3) * (34-i) / 24))) : null;
    });
    return {mode, name: item ? clip(item.t,22) : fallback + " · 暂无数据", floor: item?.f ?? 0,
      ev: item?.eventIntensity ?? (demo ? s : null), em: item?.emotion ?? (demo ? em : null),
      rule: item ? "当前地板 " + round(item.f || 0) + " · 右侧为本地推演" : "当前来源没有该类数据",
      det:track(s,14), imp:im ? track(im,6) : null, emo:track(em,9)};
  });
  const reference = byStrength[0];
  const related = new Map((reference?.lk || []).filter(link => link[1] === 0).map(link => [link[0],link[2]]));
  const candidates = byStrength.slice(0,15).map((item,index) => ({
    t:clip(item.t,22), pool:item.label, sim:item.i === reference?.i ? 1 : Math.max(0, related.get(item.i) ?? 0),
    str:item.s / 100, emo:item.em / 100, win:item.sc, base:index+1
  })).slice(0,5);

  const recall = payload.local_recall || (demo ? (() => {
    const days = Array.from({length:24}, (_,i) => [DATE(23-i), 90+(i*53)%240, 25+(i*19)%90, 14+(i*11)%60]);
    const peak = days.reduce((a,b) => a[1] >= b[1] ? a : b);
    return {days, total:sum(days.map(x=>x[1])), n:days.length, longest:days.length, peak:peak[1], peak_day:peak[0]};
  })() : null);
  const charRules = [details.budgets?.long ?? 3000, details.budgets?.short ?? 1500,
    details.budgets?.companion ?? 8000, details.budgets?.compact ?? 1500];
  const limits = {desktop: 131072, headless: 192000, web:128000};
  const profiles = Object.fromEntries(Object.entries(limits).map(([name, win], index) => {
    const lengths = visible.slice(-24).map(item => chars(item.t));
    const count = Math.max(24 - lengths.length, 0);
    const portion = [1,.7,.5][index];
    const ratio = (value, cap) => value == null ? null : Math.min(100, round(value * portion / cap * 100));
    const charsSeries = Array(count).fill(null).concat(lengths);
    const dshPct = charsSeries.map((_,i) => demo ? 12 + (i * 17 + index * 12) % 68 : null);
    return [name, {label: name + (demo ? " · 合成预算" : " · 本地预算"), win, curTok: demo ? round((dshPct.at(-1) || 0) * win / 100) : null,
      dshPct, memPct:charsSeries.map(v=>ratio(v,charRules[0] + charRules[1])),
      iniPct:charsSeries.map(v=>ratio(v,charRules[2])), sizeKB:"—"}];
  }));

  function preview(body) {
    const cap = 120, body_cap = 168, ttl = 24;
    const fields = [clip(body.topic,44), clip(body.emotion,24), clip(body.pending_tail,44)];
    const trivia = (Array.isArray(body.trivia) ? body.trivia : []).slice(0,3).map(t => clip(t,30)).filter(Boolean);
    const used = () => sum(fields.map(chars)) + sum(trivia.map(chars));
    while (trivia.length && used() > cap) trivia.pop();
    if (used() > cap) fields[2] = clip(fields[2], Math.max(0, chars(fields[2]) - (used() - cap)));
    const state = num(body.age_hours) > ttl ? "expired" : body.is_intimate ? "protected" : "ready";
    const trigger = ["rotate", "cross_day", "compress"].includes(body.trigger) ? body.trigger : "rotate";
    const prefix = "<!--MEM_START-->\n", suffix = "\n<!--MEM_END-->";
    const message = "[上次交接单 · " + trigger + "]\n" + fields.filter(Boolean).join("\n") + "\n" + trivia.join("、");
    const marker = state === "ready" ? prefix + clip(message, body_cap-prefix.length-suffix.length) + suffix : "";
    return {ttl, cap, body_cap, state, marker, used:used(), body_chars:chars(marker),
      record:{topic:fields[0], emotion:fields[1], pending_tail:fields[2], trivia, trigger}};
  }

  function counts() {
    const result = {};
    for (const card of cards) result[card.st] = (result[card.st] || 0) + 1;
    return result;
  }

  function currentCards() {
    const live = details.cards;
    const localCounts = counts();
    for (const [status, value] of Object.entries(live?.counts || {})) {
      localCounts[status] = (localCounts[status] || 0) + value;
    }
    return {
      cards: [...cards, ...(live?.cards || []).map(card => ({ ...card, readOnly: !canWrite("review") || !card.version }))], counts: localCounts,
      batches: (live?.batches ?? 0) + session.batches, capacity: live?.capacity ?? 300,
      model: live?.model ?? "本地规则", history: live?.history?.length ? live.history : history,
      skills: (live?.skills || []).map(skill => ({ ...skill, readOnly: !canWrite("outcome") || !skill.version })), vault: details.vault
    };
  }

  function distill(limit) {
    if (limit <= 0) return 0;
    let n = 0;
    for (const item of byStrength) {
      const key = item.t.normalize("NFKC").toLowerCase().replace(/[\s\p{P}]/gu,"");
      if (!key || added.has(key)) continue;
      added.add(key);
      const kind = item.en[0] ? "实体摘要" : "正文摘要";
      cards.push({
        id:"L-"+(cards.length+1), version:1, st:"provisional", kind,
        txt:clip(item.t,60), full:clip(item.t,160), quote:"",
        model:"本地规则", confLabel:"规则候选", str:item.s,
        note:"内存预演候选", updatedAt:DATE(0), sourceRevs:"快照 #"+item.id
      });
      if (++n >= limit) break;
    }
    return n;
  }

  async function audit() {
    const byPool = new Map();
    for (const item of snapshot.items) {
      if (!byPool.has(item.label)) byPool.set(item.label, []);
      byPool.get(item.label).push(item);
    }
    const encoder = new TextEncoder();
    const pools = [];
    for (const [name, items] of details.pools ? [] : byPool) {
      const bytes = encoder.encode(JSON.stringify(items));
      const sha = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",bytes)), b=>b.toString(16).padStart(2,"0")).join("");
      pools.push({name,desc:"快照分组",count:items.length,unit:"条",kb:round(bytes.length/1024),sha:sha.slice(0,16)+"…"+sha.slice(-8),status:"已计算"});
    }
    const checks = [
      ["节点与记录 1:1", snapshot.items.length === payload.records.length],
      ["节点 ID 唯一", new Set(snapshot.items.map(item=>item.id)).size === payload.records.length],
      ["连线索引有效", snapshot.items.every(item=>item.lk.every(([i])=>i>=0 && i<snapshot.items.length))],
      ["坐标为有限数值", snapshot.items.every(item=>item.pos.every(Number.isFinite))],
      ["私密记录内容隐藏", payload.records.every(item=>!item.private || item.content === "私密记忆 · 内容已隐藏")],
      ["原始向量未进入可视化", payload.records.every(item=>!Object.hasOwn(item,"embedding"))],
      ["分组计数一致", sum(pools.map(pool=>pool.count)) === payload.records.length]
    ];
    const items = checks.map(([name,ok])=>({name,ok,ev:ok ? "本地校验通过" : "本地校验失败"}));
    const remote = await readSourceModule("audit").catch(()=>null);
    for (const row of (remote||[]).slice(0,4)) items.push({name:row.label,ev:row.value,ok:null});
    return {pools:details.pools || pools, items, total:checks.length, passed:checks.filter(x=>x[1]).length,
      max_delta:0, all_unchanged:true, checked_at:new Date().toLocaleTimeString()};
  }

  async function request(path, options={}) {
    const method = options.method || "GET";
    const body = options.body ? JSON.parse(options.body) : {};
    const out = (value, status=200) => new Response(JSON.stringify(value), {status,headers:{"Content-Type":"application/json"}});
    const action = writableRoutes[path];
    if (method === "POST" && canWrite(action) && !(action === "review" && !String(body.id).startsWith("source-"))) {
      try { return out(await mutate(action, body)); }
      catch (error) { return out({error:error.message}, 409); }
    }
    switch (path) {
      case "/api/settings":
        if (method === "POST") session.values = Object.fromEntries(Object.entries(body.values || {}).filter(([,v])=>Number.isFinite(v)));
        return out({defaults:details.settings?.defaults, values:{...details.settings?.values, ...session.values},
          revision:write.revisions?.["state/runtime_overrides.json"]});
      case "/api/telemetry":
        return out({recall, budgets:details.budgets});
      case "/api/dsh":
        const dsh = session.dsh ?? details.dsh ?? {enabled:true, dryRun:true, minTools:6, revisions:{}};
        if (method === "POST") {
          const vacuum = body.updates?.["dsh-session-vacuum/config.json"] || {};
          const tools = num(body.updates?.["cordis.patch.yml"]?.minTools);
          session.dsh = {enabled:vacuum.enabled === true,dryRun:vacuum.dryRun === true,minTools:tools >= 1 && tools <= 64 ? tools : dsh.minTools,revisions:{}};
        }
        const revisions = Object.fromEntries(Object.entries(write.revisions || {}).filter(([name]) => !name.startsWith("state/")));
        return out({...session.dsh ?? dsh, revisions});
      case "/api/dsh/probe":
        return out({error:"来源文件只读"},403);
      case "/api/handover/preview":
        return out(preview(body));
      case "/api/handover":
        if (method === "POST") {
          const view = preview(body);
          session.handover = view.state === "ready" ? {...view.record,seq:++session.seq} : null;
          return out({seq:session.seq,protected:view.state === "protected"});
        }
        return out({record:Object.hasOwn(session, "handover") ? session.handover : details.handover,
          revision:write.revisions?.["state/handover.json"]});
      case "/api/cards":
        return out(currentCards());
      case "/api/cards/generate": {
        const requested = Math.min(5, Math.max(1, num(body.max_batches) || 1));
        const added = distill(Math.min(Math.max(0,300-cards.length), requested*6));
        session.batches += added ? 1 : 0;
        history.push([cards.length,counts().active || 0,counts().stale || 0]);
        return out({planned:added,sources:visible.length,pending:counts().provisional||0,batches_remaining:0});
      }
      case "/api/cards/review": {
        const card = cards.find(entry => entry.id === body.id);
        if (!card || card.version !== body.version) return out({error:"候选已更新"},409);
        card.st = body.approved ? "active" : "rejected";
        card.version++;
        history.push([cards.length,counts().active || 0,0]);
        return out({status:card.st});
      }
      case "/api/cards/maintain":
        return out({current:counts().active||0,skills:0,skill_capacity:30,stale:0,archived:0});
      case "/api/cards/outcome":
        return out({status:"sandbox"});
      case "/api/vault/probe":
        return out({guard_ok:true,preserved_last_distill:details.vault?.last || "沙盒未写入"});
      case "/api/audit":
      case "/api/audit/run":
        return out(await audit());
      default:
        return out({error:"本地计算未提供该操作"},404);
    }
  }

  if (demo && !session.initialized) {
    for (let i=0;i<4;i++) {
      if (distill(6)) session.batches++;
      if (cards.length > i) cards[i].st = "active";
      history.push([cards.length, counts().active || 0, 0]);
    }
    session.handover = {topic:"潮汐图册观测方案",emotion:"专注",pending_tail:"校对潮位表",trivia:["复核观测日期"],trigger:"rotate"};
  }
  session.initialized = true;
  return {samples,candidates,profiles,recall,request,demo,canWrite,undo:(body = {}) => mutate("undo", body), undoStatus:module => write.undo?.[module]};
};

