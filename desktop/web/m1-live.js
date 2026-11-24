// Keep the M1 layout anchored to one data connection.
(() => {
  const gapStart = 3.75, gapEnd = 4.22;

  function visual(item) {
    return { p: item.p, sc: item.sc, s: item.s, rc: item.rc };
  }

  function anchor(items) {
    return {
      items: items.map(item => ({ ...item, pos: flipped(item.pos), appearance: visual(item), edgeEntities: item.en })),
      added: []
    };
  }

  function flipped([x, y, z]) {
    return [x, y, z];
  }

  function placement(item, incoming, pinned) {
    const position = flipped(item.pos);
    // Use the local displacement from a related, already placed star.
    for (const [index] of item.lk) {
      const neighbor = incoming[index];
      const old = neighbor && pinned.get(neighbor.key);
      if (!old || neighbor.group !== item.group) continue;
      const now = flipped(neighbor.pos);
      const shift = old.pos.map((value, axis) => value - now[axis]);
      const weight = Math.min(0.65, 0.20 / (Math.hypot(...shift) || 1));
      return position.map((value, axis) => value + shift[axis] * weight);
    }
    return position;
  }

  function merge(scene, snapshot) {
    const incoming = snapshot.items;
    const byKey = new Map(incoming.map(item => [item.key, item]));
    const pinned = new Map(scene.items.map(item => [item.key, item]));
    const kept = scene.items.filter(item => byKey.has(item.key));
    const added = incoming.filter(item => !pinned.has(item.key));
    const order = [...kept.map(item => byKey.get(item.key)), ...added];
    const indexByKey = new Map(order.map((item, index) => [item.key, index]));
    const resolve = (items, index) => indexByKey.get(items[index]?.key);
    const items = order.map((item, index) => {
      const old = pinned.get(item.key);
      const samePrivacy = old && (old.k === '私密') === (item.k === '私密');
      const linkRows = samePrivacy ? old.lk : item.lk;
      const linkInput = samePrivacy ? scene.items : incoming;
      const lk = linkRows.flatMap(([target, kind, weight]) => {
        const resolved = resolve(linkInput, target);
        return resolved === undefined ? [] : [[resolved, kind, weight]];
      });
      return {
        ...item, i: index, id: String(index + 1), lk,
        pos: old?.pos ?? placement(item, incoming, pinned),
        appearance: samePrivacy ? old.appearance : visual(item),
        edgeEntities: samePrivacy ? old.edgeEntities : item.en
      };
    });
    return { items, added: items.slice(kept.length), fit: scene.fit };
  }

  function arrivalPoint(destination, progress, time, lane = 0) {
    const angle = (gapStart + gapEnd) / 2 + lane + time * (0.032 - 0.018);
    const x = Math.cos(angle), z = Math.sin(angle);
    const crossing = 0.36, gateRadius = 1.32;
    if (progress <= crossing) {
      const radius = 2.12 - (2.12 - gateRadius) * progress / crossing;
      return [x * radius, 0.03, z * radius];
    }
    const n = Math.min(1, (progress - crossing) / (1 - crossing));
    const eased = n * n * (3 - 2 * n);
    return [
      x * gateRadius * (1 - eased) + destination[0] * eased,
      0.03 * (1 - eased) + destination[1] * eased + Math.sin(Math.PI * eased) * 0.18,
      z * gateRadius * (1 - eased) + destination[2] * eased
    ];
  }

  window.M1Live = { anchor, merge, arrivalPoint, gapStart, gapEnd };
})();

