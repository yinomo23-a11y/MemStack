(() => {
  const cards = [...document.querySelectorAll("[data-endpoint]")];
  const pending = new Map();
  for (const card of cards) {
    const endpoint = card.dataset.endpoint;
    const section = card.dataset.section || "";
    const body = card.querySelector(".remote-rows");
    if (!pending.has(endpoint)) pending.set(endpoint, window.parent.readModule(endpoint));
    pending.get(endpoint).then(rows => {
      if (rows === null) {
        body.textContent = "未接入";
        return;
      }
      const results = rows.filter(row => row.section === section);
      body.replaceChildren();
      if (!results.length) body.textContent = "暂无数据";
      for (const { label, value } of results) {
        const row = document.createElement("div");
        const tag = document.createElement("strong");
        const detail = document.createElement("span");
        tag.textContent = label;
        detail.textContent = value;
        row.append(tag, detail);
        body.append(row);
      }
    }).catch(error => { body.textContent = error.message || "读取失败"; });
  }
})();

