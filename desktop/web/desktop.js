(() => {
  const $ = id => document.getElementById(id);
  const stage = $("stage");
  const dialog = $("sourceDialog");
  const tabs = [...document.querySelectorAll("#navTabs .tab")];
  let profile = { kind: "empty" };
  let reconnectTarget = null;
  let reconnectTimer;
  let offline = false;
  let capabilities = new Set();
  let active = "m1";
  let busy = false;
  let refreshing = false;
  let recordsKey = "";
  let matrixKey = "";
  let labSession = {};
  let mapping = null;
  let api;

  function message(text) {
    $("sourceError").textContent = text;
  }

  function setBusy(value) {
    busy = value;
    for (const button of dialog.querySelectorAll("button, select")) button.disabled = value;
    $("btnRefresh").disabled = value || refreshing;
    $("btnUndo").disabled = value || refreshing;
    $("btnSource").disabled = value;
    $("btnDemo").disabled = value;
    $("emptyDemo").disabled = value;
    $("emptySource").disabled = value;
  }

  async function task(action, button, label) {
    if (busy) return;
    setBusy(true);
    const finish = button && window.showButtonProgress(button, label);
    message("");
    try {
      await action();
    } catch (error) {
      message(error.message || "操作失败，请重试。");
      if (!dialog.open) dialog.showModal();
    } finally {
      finish?.();
      setBusy(false);
    }
  }

  function unwrap(reply) {
    if (!reply.ok) throw new Error(reply.error);
    return reply.data;
  }

  function show(mod, preserveView = false) {
    const view = preserveView && !stage.hidden ? stage.contentWindow?.captureView?.() : null;
    active = mod;
    window.restoredView = view ? { mod, ...view } : null;
    for (const tab of tabs) tab.classList.toggle("active", tab.dataset.mod === mod);
    const graph = ["m1", "m2", "m3"].includes(mod);
    const empty = profile.kind === "empty" || (graph && mod !== "m1" && !window.consoleSnapshot?.items.length);
    $("workspaceEmpty").hidden = !empty;
    stage.hidden = empty;
    if (empty) {
      stage.removeAttribute("src");
      $("emptyTitle").textContent = profile.kind === "empty" ? "选择记忆数据" : "这份数据还没有记忆";
      $("emptyDetail").textContent = profile.kind === "empty" ? "接入自己的数据，或打开合成演示。" : "添加记忆后会在下次刷新显示。";
      return;
    }
    stage.src = mod + ".html";
  }

  window.readModule = async endpoint => {
    if (!capabilities.has(endpoint)) return null;
    return unwrap(await api.module(endpoint));
  };

  function publish(payload, reset, undoneAction) {
    window.m1Connection = payload.connection;
    capabilities = new Set(payload.capabilities.features);
    const nextRecords = JSON.stringify([payload.records, payload.capabilities.pools, payload.capabilities.strength_unit]);
    const graphChanged = reset || nextRecords !== recordsKey;
    if (graphChanged) window.consoleSnapshot = projectSnapshot(payload);
    const cards = payload.details?.cards;
    const summary = {
      distill: cards ? { cards: cards.cards.length, active: cards.counts.active ?? 0, batches: cards.batches } : null,
      audit: payload.local_recall ? { total_hits: payload.local_recall.total } : null,
      runtime: payload.details?.runtime ?? null
    };
    const nextMatrix = JSON.stringify(summary);
    const matrixChanged = nextMatrix !== matrixKey;
    if (graphChanged || matrixChanged) window.geminiSnapshot = { ...projectGemini(window.consoleSnapshot), ...summary };
    recordsKey = nextRecords;
    matrixKey = nextMatrix;
    window.labRuntime = createLabRuntime(window.geminiSnapshot, payload, profile.kind === "demo", window.readModule, labSession,
      (action, body) => mutate(payload.connection, action, body),
      async body => unwrap(await api.ai_review(payload.connection, body)));
    $("btnUndo").hidden = !payload.write?.actions?.length;
    const sourceNames = { empty: "未连接", demo: "合成演示", folder: "忆栈目录", sqlite: "SQLite", jsonl: "JSONL", http: "HTTP 接口" };
    $("navTotal").textContent = profile.kind === "empty" ? "未连接" : sourceNames[profile.kind] + " · " + payload.records.length.toLocaleString() + " 条" + (payload.truncated ? "（已截取）" : "");
    $("navLock").textContent = profile.kind === "empty" ? "只读" : profile.kind === "demo" ? "合成数据" : "每 30 秒刷新";
    $("navLock").title = profile.kind === "empty" ? "" : "上次刷新 " + new Date().toLocaleTimeString();
    const graph = ["m1", "m2", "m3"].includes(active);
    if (!reset && graphChanged && active === "m1" && !stage.hidden && stage.contentWindow?.updateM1Snapshot) {
      stage.contentWindow.updateM1Snapshot(window.geminiSnapshot);
    } else if (reset || (graph && (graphChanged || (active === "m2" && matrixChanged)))) {
      show(active, !reset);
    } else if (!graph && !stage.hidden) {
      stage.contentWindow?.refreshData?.(undoneAction);
    }
  }

  async function connect(next) {
    const payload = unwrap(await api.connect(next));
    profile = next;
    reconnectTarget = next.kind === "empty" || next.kind === "demo" ? null : next;
    offline = false;
    clearTimeout(reconnectTimer);
    labSession = {};
    window.m1Scene = null;
    publish(payload, true);
    if (dialog.open) dialog.close();
  }

  function retryLater(error) {
    offline = true;
    $("navLock").textContent = "重连中";
    $("navLock").title = error.message || "保留上次快照，5 秒后重试。";
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(() => refresh(true), 5000);
  }

  async function mutate(connection, action, body) {
    if (busy) throw new Error("另一项操作进行中。");
    setBusy(true);
    try {
      const result = unwrap(await api.mutate(connection, action, body));
      if (result.written || result.undone) {
        try { publish(unwrap(await api.refresh()), false, result.undone ? result.action : undefined); }
        catch (error) { retryLater(error); }
      }
      return result;
    } finally {
      setBusy(false);
    }
  }

  async function refresh(manual = false) {
    if (!api || refreshing || (!reconnectTarget && profile.kind !== "demo") || (!manual && profile.kind === "demo")) return;
    if (busy) { if (offline) retryLater(new Error("操作完成后重试。")); return; }
    const current = profile;
    refreshing = true;
    $("btnRefresh").disabled = true;
    $("btnUndo").disabled = true;
    try {
      if (profile.kind === "empty" && reconnectTarget) await connect(reconnectTarget);
      else {
        const payload = unwrap(await api.refresh());
        if (current === profile) publish(payload, false);
      }
      offline = false;
      clearTimeout(reconnectTimer);
    } catch (error) {
      if (current === profile) {
        retryLater(error);
      }
    } finally {
      refreshing = false;
      $("btnRefresh").disabled = busy;
      $("btnUndo").disabled = busy;
    }
  }

  function choose(kind) {
    $("urlForm").hidden = kind !== "http";
    $("mapForm").hidden = kind !== "mapping";
  }

  function openDialog() {
    choose("");
    message("");
    $("folderExtras").hidden = profile.kind !== "folder";
    $("toggleWrite").textContent = profile.write_enabled ? "关闭本机写入" : "启用本机写入";
    dialog.showModal();
  }

  async function inspect(table = "") {
    const meta = unwrap(await api.inspect(mapping.kind, mapping.path, table));
    mapping.table = meta.table;
    const tables = $("tableSelect");
    tables.replaceChildren(...meta.tables.map(name => new Option(name, name)));
    tables.value = meta.table;
    $("tableLabel").hidden = mapping.kind !== "sqlite";
    const fields = $("fields");
    fields.replaceChildren(...meta.fields.map(name => new Option(name, name)));
    for (const [key, guess] of Object.entries(meta.guesses)) $("map-" + key).value = guess;
    $("fileName").textContent = mapping.path.split(/[\\/]/).at(-1);
    choose("mapping");
  }

  for (const tab of tabs) tab.addEventListener("click", () => show(tab.dataset.mod));
  $("btnSource").addEventListener("click", openDialog);
  $("emptySource").addEventListener("click", openDialog);
  for (const id of ["btnDemo", "emptyDemo"]) {
    $(id).addEventListener("click", event => task(() => connect({ kind: "demo" }), event.currentTarget, "加载中…"));
  }
  $("btnRefresh").addEventListener("click", async event => {
    if (!api || busy || refreshing || (!reconnectTarget && profile.kind !== "demo")) return;
    const finish = window.showButtonProgress(event.currentTarget, "刷新中…");
    try { await refresh(true); }
    finally { finish(); }
  });
  $("btnUndo").addEventListener("click", async event => {
    if (!confirm("撤销当前数据源的上一次实际写入？")) return;
    const finish = window.showButtonProgress(event.currentTarget, "撤销中…");
    try {
      const result = await window.labRuntime.undo();
      $("navLock").textContent = "已撤销";
      $("navLock").title = "已撤销：" + result.undone;
    } catch (error) {
      $("navLock").textContent = "撤销未执行";
      $("navLock").title = error.message;
      message(error.message);
      if (!dialog.open) dialog.showModal();
    } finally {
      finish();
    }
  });
  $("sourceClose").addEventListener("click", () => dialog.close());
  dialog.addEventListener("cancel", event => { if (busy) event.preventDefault(); });
  $("pickFolder").addEventListener("click", event => task(async () => {
    const path = await api.pick("folder");
    if (path) await connect({ kind: "folder", path });
  }, event.currentTarget, "选择中…"));
  for (const [button, key] of [["pickDsh", "dsh_path"], ["pickVault", "vault_path"]]) {
    $(button).addEventListener("click", event => task(async () => {
      const path = await api.pick("folder");
      if (path) await connect({ ...profile, [key]: path });
    }, event.currentTarget, "选择中…"));
  }
  $("toggleWrite").addEventListener("click", event => task(async () => {
    if (!profile.write_enabled && !confirm("启用后，保存操作会调用此目录中的本机记忆模型。仅为你信任的记忆模型目录启用。")) return;
    const next = { ...profile, write_enabled: !profile.write_enabled };
    await connect(next);
    $("toggleWrite").textContent = next.write_enabled ? "关闭本机写入" : "启用本机写入";
  }, event.currentTarget, "切换中…"));
  $("pickFile").addEventListener("click", event => task(async () => {
    const path = await api.pick("file");
    if (!path) return;
    mapping = { path, kind: /\.(jsonl|ndjson)$/i.test(path) ? "jsonl" : "sqlite" };
    await inspect();
  }, event.currentTarget, "选择中…"));
  $("pickHttp").addEventListener("click", () => {
    message("");
    choose("http");
    $("httpUrl").value = profile.kind === "http" ? profile.url : "";
    $("httpUrl").focus();
  });
  $("disconnect").addEventListener("click", event => task(() => connect({ kind: "empty" }), event.currentTarget, "断开中…"));
  $("tableSelect").addEventListener("change", event => task(() => inspect(event.target.value)));
  $("urlForm").addEventListener("submit", event => {
    event.preventDefault();
    task(() => connect({ kind: "http", url: $("httpUrl").value }), event.submitter, "连接中…");
  });
  $("mapForm").addEventListener("submit", event => {
    event.preventDefault();
    const fields = Object.fromEntries([...event.target.querySelectorAll("[data-field]")].map(input => [input.dataset.field, input.value]));
    task(() => connect({ ...mapping, mapping: fields }), event.submitter, "连接中…");
  });
  document.addEventListener("keydown", event => {
    if (dialog.open || event.ctrlKey || event.altKey || event.metaKey || !/^[1-6]$/.test(event.key)) return;
    show("m" + event.key);
  });

  show(active);
  function ready() {
    if (api) return;
    api = window.pywebview.api;
    api.saved().then(saved => {
      reconnectTarget = saved;
      if (saved) void refresh(true);
    });
    window.setInterval(async () => {
      if (busy || refreshing || !await api.reconnect_requested()) return;
      const saved = await api.saved();
      if (saved && (profile.kind === "demo" || !reconnectTarget)) reconnectTarget = saved;
      if (profile.kind === "demo" && saved) await connect(saved).catch(retryLater);
      else if (saved) await refresh(true);
      else if (!dialog.open) openDialog();
    }, 1000);
  }
  if (window.pywebview?.api) ready();
  else window.addEventListener("pywebviewready", ready, { once: true });
  window.setInterval(refresh, 30_000);
})();

