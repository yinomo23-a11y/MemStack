// Prepare the original console's graph input from the public record protocol.
window.projectSnapshot = function (snapshot) {
  const { records, capabilities } = snapshot;
  const pools = new Map(capabilities.pools.map(pool => [pool.id, pool.label]));
  const knownSlots = ["active", "shortterm", "intimate", "cold", "working"];
  const labels = Array(5).fill("");
  const slot = new Map();
  const native = [...pools.keys()].every(key => knownSlots.includes(key));
  const poolIds = [...pools.keys()];
  poolIds.forEach((key, index) => {
    const p = native ? knownSlots.indexOf(key) : Math.min(index, 4);
    slot.set(key, p);
    labels[p] = !native && p === 4 && poolIds.length > 5 ? "其他分组" : pools.get(key);
  });

  const strengths = records.flatMap(row => row.strength === null ? [] : [row.strength]);
  const min = strengths.length ? Math.min(0, ...strengths) : 0;
  const max = strengths.length ? Math.max(1, ...strengths) : 100;
  const normalize = value => value === null ? 0 : 100 * (value - min) / (max - min);
  const poolCounts = Array(5).fill(0);
  const towerCounts = Array(6).fill(0);
  const towerLabels = Array(6).fill("");
  const activeTiers = { L0: 0, L1: 0, L2: 0, episode: 0, schedule: 0 };
  const shorttermTiers = { fresh: 0, decay: 0, forget: 0, tool: 0 };
  const floorForPool = [3, 4, 1, 5, 0];
  labels.forEach((label, p) => { towerLabels[floorForPool[p]] = label; });
  // The active pool splits across two floors by its decay floor, exactly as the
  // console's snapshot does (floor >= 45 → core/episode floor, else long-term).
  // Without the split, floor 2 stays empty and the tower visibly loses a ring.
  const ACTIVE_SPLIT_FLOOR = 45;
  if (labels[0]) {
    towerLabels[2] = "45 核心/情景地板";
    towerLabels[3] = "长期活跃";
  }
  const clusters = new Map();
  const lastInCluster = new Map();
  let edges = 0;
  const now = Date.now();

  const items = records.map((row, i) => {
    const p = slot.get(row.pool) ?? 0;
    // Split active across floors 2/3 by its decay floor; other pools map 1:1.
    const schedule = Boolean(row.schedule);
    const tw = schedule ? 0 : p === 0 && (row.floor ?? 0) >= ACTIVE_SPLIT_FLOOR ? 2 : floorForPool[p];
    const s = normalize(row.strength);
    // When the source carries a position (the SVD of its embedding, computed in
    // Python and never leaving as a vector), the cosmos keeps its real shape.
    // The golden-angle spiral below is only the fallback for sources that
    // cannot supply one, so those still get a layout instead of a pile at 0,0,0.
    const placed = Array.isArray(row.pos) && row.pos.length === 3;
    const theta = i * 2.39996;
    const tilt = Math.sin(i * 1.618) * 0.4;
    const radius = [0.32, 0.58, 0.52, 1.15, 0.14][p] + (1 - s / 100) * 0.53;
    const lk = [];
    for (const name of row.entities) {
      clusters.set(name, (clusters.get(name) || 0) + 1);
      const previous = lastInCluster.get(name);
      if (previous !== undefined && lk.length < 4 && !lk.some(([id]) => id === previous)) {
        lk.push([previous, 2, 1]);
        edges++;
      }
      lastInCluster.set(name, i);
    }
    poolCounts[p]++;
    towerCounts[tw]++;
    const ageDays = row.time ? Math.max(0, (now - Date.parse(row.time)) / 86400000) : 0;
    const level = row.level;
    const floor = row.floor;
    const episode = Boolean(row.episode);
    if (p === 0) {
      if (ageDays <= 7) activeTiers.L0++;
      else if (ageDays <= 30) activeTiers.L1++;
      else activeTiers.L2++;
      if (episode) activeTiers.episode++;
    }
    if (schedule) {
      activeTiers.schedule++;
      towerLabels[0] = "课表/工作窗";
    }
    if (p === 1) {
      if (s >= 80) shorttermTiers.fresh++;
      else if (s >= 30) shorttermTiers.decay++;
      else shorttermTiers.forget++;
    }
    return {
      i, id: String(i + 1), key: row.key, p, tw, s, lk,
      cosmosLinks: row.cosmos_links || null,
      cosmosEntities: row.cosmos_entities ?? null,
      visualAge: row.visual_age ?? null,
      visualFloor: row.visual_floor ?? null,
      impression: row.impression ?? null,
      emotion: row.emotion ?? null,
      eventIntensity: row.event_intensity ?? null,
      en: row.entities, tg: row.tags || [],
      pos: placed
        ? row.pos
        : [Math.cos(theta) * radius, tilt * radius, Math.sin(theta) * radius],
      label: pools.get(row.pool) || row.pool, group: row.pool,
      t: row.content || "（空内容）",
      strength: row.strength,
      strengthLabel: row.strength === null ? "—" : row.strength.toLocaleString() + capabilities.strength_unit,
      time: row.time,
      ag: ageDays,
      k: row.private ? "私密" : (row.kind || "记忆"),
      mt: row.kind || "",
      // null means the source did not carry the input; the matrix shows that as unknown.
      f: floor === null || floor === undefined ? 0 : floor,
      fKnown: floor !== null && floor !== undefined,
      lv: level === null || level === undefined ? 0 : level,
      ep: episode ? 1 : 0,
      sc: schedule ? 1 : 0,
      im: 0, em: row.emotion || 0,
      mc: row.mentions || 0, rc: row.recalls || 0,
    };
  });

  return {
    items, pools: poolCounts, poolLabels: labels, tower_tiers: towerCounts, towerLabels,
    active_tiers: activeTiers, shortterm_tiers: shorttermTiers,
    groups: capabilities.pools,
    entities: [...clusters.entries()].sort((a, b) => b[1] - a[1]),
    edge_counts: { total: edges },
    // The folder/API protocol does not carry distillation, audit or runtime telemetry.
    // These stay null so the matrix can label them instead of inventing numbers.
    distill: null, audit: null, runtime: null,
    generated_at: new Date().toLocaleString(),
  };
};

window.escapeHtml = function (value) {
  return String(value).replace(/[&<>"']/g, char =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
};


// Adapt the normalized source to the fields expected by Gemini's own renderers.
window.projectGemini = function (snapshot) {
  const slots = ["active", "shortterm", "intimate", "cold", "working"];
  const previous = new Map();
  const clusters = new Map();
  const items = snapshot.items.map(item => {
    const en = item.cosmosEntities ?? item.en;
    const fallback = [];
    for (const name of en) {
      clusters.set(name, (clusters.get(name) || 0) + 1);
      if (previous.has(name) && !fallback.length) fallback.push([previous.get(name), 2, 1]);
      previous.set(name, item.i);
    }
    return {
      ...item, en,
      lk: item.cosmosLinks ?? (item.cosmosEntities === null ? item.lk : fallback),
      s: item.visualAge !== null && item.strength !== null
        ? Math.max(0, Math.min(100, item.strength)) : item.s,
      f: item.visualFloor ?? item.f, im: item.impression ?? item.im, em: item.emotion ?? item.em,
      ag: item.visualAge ?? item.ag
    };
  });
  const active_tiers = { L0: 0, L1: 0, L2: 0, episode: 0, schedule: 0 };
  for (const item of items) {
    if (item.p === 0) {
      active_tiers[item.ag <= 7 ? "L0" : item.ag <= 30 ? "L1" : "L2"]++;
      if (item.ep) active_tiers.episode++;
    }
    if (item.sc) active_tiers.schedule++;
  }
  return {
    ...snapshot, items, active_tiers,
    pools: Object.fromEntries(slots.map((name, index) => [name, snapshot.pools[index]])),
    entities: [...clusters.entries()].sort((a, b) => b[1] - a[1]),
    edge_counts: { total: items.reduce((n, item) => n + item.lk.length, 0) },
    distill: { cards: 0, batches: 0, active: 0 },
    audit: { total_hits: 0 }, runtime: {}
  };
};
