const state = {
  templates: [],
  currentTemplate: null,
  controller: null,
  structuredText: "",
  realisticText: "",
  tokenToOriginal: {},
  tokenToFake: {},
  fakeToToken: {},
};

const $ = (id) => document.getElementById(id);
const DEFAULT_TEMPLATE_ID = "default-pii-v1";
const TEMPLATE_DISPLAY_NAMES = {
  "default-pii-v1": "default-pii-v1-llama-7B-q4",
};

function ensureDefaultTemplateSelection() {
  const selectors = ["template-select", "editor-template-select", "map-template-select"];
  for (const id of selectors) {
    const sel = $(id);
    if (!sel) continue;
    if (sel.querySelector(`option[value="${DEFAULT_TEMPLATE_ID}"]`)) {
      sel.value = DEFAULT_TEMPLATE_ID;
    }
  }
}

function logTimeline(msg) {
  const timeline = $("timeline");
  if (!timeline) return;
  const el = document.createElement("div");
  el.className = "timeline-item";
  el.textContent = `${new Date().toLocaleTimeString()}  ${msg}`;
  timeline.prepend(el);
}

function setStatus(id, text, ok = true) {
  const el = $(id);
  if (!el) return;
  el.textContent = text;
  el.className = `status ${ok ? "ok" : "bad"}`;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return res.json();
  }
  return res.text();
}

async function loadTemplates() {
  const data = await api("/v2/templates");
  state.templates = data.templates || [];
  const selects = ["template-select", "editor-template-select", "map-template-select"];
  for (const id of selects) {
    const sel = $(id);
    if (!sel) continue;
    sel.innerHTML = "";
    state.templates.forEach((t) => {
      const op = document.createElement("option");
      op.value = t.template_id;
      const label = TEMPLATE_DISPLAY_NAMES[t.template_id] || t.template_id;
      op.textContent = `${label} (v${t.version})`;
      sel.append(op);
    });
    if (state.templates.some((t) => t.template_id === DEFAULT_TEMPLATE_ID)) {
      sel.value = DEFAULT_TEMPLATE_ID;
    }
  }
  if ($("kpi-template")) {
    const preferred = state.templates.find((t) => t.template_id === DEFAULT_TEMPLATE_ID);
    $("kpi-template").textContent = (preferred || state.templates[0] || {}).template_id || "-";
  }
  ensureDefaultTemplateSelection();
}

function switchTab(tab) {
  document.querySelectorAll(".tab").forEach((el) => {
    el.classList.toggle("active", el.dataset.tab === tab);
  });
  document.querySelectorAll(".panel").forEach((el) => {
    el.classList.toggle("active", el.id === `panel-${tab}`);
  });
}

function renderMappingTable() {
  const table = $("mapping-table");
  if (!table) return;
  const tbody = table.querySelector("tbody");
  tbody.innerHTML = "";
  const tokens = Object.keys(state.tokenToOriginal);
  tokens.forEach((token) => {
    const tr = document.createElement("tr");
    const tokenCell = document.createElement("td");
    tokenCell.textContent = token;
    const originalCell = document.createElement("td");
    originalCell.textContent = state.tokenToOriginal[token] || "";
    const realisticCell = document.createElement("td");
    realisticCell.textContent = state.tokenToFake[token] || "";
    tr.append(tokenCell, originalCell, realisticCell);
    tbody.append(tr);
  });

  const sourceLen = $("source-text").value.length || 1;
  const protectedChars = Object.values(state.tokenToOriginal).reduce((acc, v) => acc + (v || "").length, 0);
  const pct = Math.min(100, Math.round((protectedChars / sourceLen) * 100));
  $("kpi-entities").textContent = `${tokens.length}`;
  $("kpi-coverage").textContent = `${pct}%`;
}

function resetStreamView() {
  state.structuredText = "";
  state.realisticText = "";
  state.tokenToOriginal = {};
  state.tokenToFake = {};
  state.fakeToToken = {};
  $("structured-output").textContent = "";
  $("realistic-output").textContent = "";
  $("timeline").innerHTML = "";
  renderMappingTable();
}

function refreshFakeProviderHint() {
  const provider = $("fake-provider").value;
  const note = $("fake-provider-note");
  if (provider === "llm") {
    note.textContent = "LLM fake generation enabled: better style matching, but noticeably higher latency.";
    note.className = "hint bad";
    return;
  }
  note.textContent = "Faker mode enabled: fast deterministic pseudonyms for demo throughput.";
  note.className = "hint ok";
}

async function streamShowcase() {
  resetStreamView();
  if (state.controller) {
    state.controller.abort();
  }
  state.controller = new AbortController();

  const payload = {
    session_id: $("session-id").value.trim(),
    template_id: $("template-select").value,
    text: $("source-text").value,
    fake_provider: $("fake-provider").value,
    language: $("language-select").value,
    token_delay_ms: Number($("token-delay").value || 0),
  };
  $("kpi-template").textContent = payload.template_id;
  logTimeline(`starting stream (${payload.fake_provider})...`);

  const res = await fetch("/v2/anonymize/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: state.controller.signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`Stream failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventType = "message";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const evtChunk of events) {
      const lines = evtChunk.split("\n");
      let dataStr = "";
      for (const ln of lines) {
        if (ln.startsWith("event:")) eventType = ln.slice(6).trim();
        if (ln.startsWith("data:")) dataStr += ln.slice(5).trim();
      }
      if (!dataStr) continue;
      const payloadObj = JSON.parse(dataStr);

      if (eventType === "structured_token") {
        state.structuredText = payloadObj.text_so_far;
        $("structured-output").textContent = state.structuredText;
      } else if (eventType === "realistic_token") {
        state.realisticText = payloadObj.text_so_far;
        $("realistic-output").textContent = state.realisticText;
      } else if (eventType === "structured_done") {
        state.tokenToOriginal = payloadObj.mapping || {};
        renderMappingTable();
        logTimeline("structured rendering complete");
      } else if (eventType === "realistic_done") {
        state.tokenToFake = payloadObj.token_to_fake || {};
        state.fakeToToken = payloadObj.fake_to_token || {};
        renderMappingTable();
        $("deanon-input").value = state.realisticText || state.structuredText;
        logTimeline("realistic rendering complete");
      } else if (eventType === "phase") {
        logTimeline(payloadObj.phase);
      } else if (eventType === "error") {
        logTimeline(`error: ${payloadObj.detail}`);
      } else if (eventType === "done") {
        logTimeline("stream complete");
      }
    }
  }
}

async function runDeanonymize() {
  const text = $("deanon-input").value;
  const mapping = {
    token_to_original: state.tokenToOriginal,
    token_to_fake: state.tokenToFake,
    fake_to_token: state.fakeToToken,
  };
  const res = await api("/v2/deanonymize", {
    method: "POST",
    body: JSON.stringify({ text, mapping }),
  });
  $("deanon-output").value = res.text || "";
}

function parseEditorTemplate() {
  return JSON.parse($("template-json").value);
}

function addEntityFromQuickForm() {
  const idRaw = $("quick-entity-id").value.trim();
  const instructions = $("quick-entity-instructions").value.trim();
  const positiveExample = $("quick-entity-example").value.trim();
  const provider = $("quick-entity-provider").value.trim();
  const enabled = $("quick-entity-enable").checked;
  const addToPostpass = $("quick-entity-add-postpass").checked;

  if (!idRaw || !instructions) {
    throw new Error("Entity ID and Instruction are required.");
  }

  const id = idRaw.toUpperCase().replace(/[^A-Z0-9_]/g, "_");
  const tmpl = parseEditorTemplate();
  tmpl.entities = tmpl.entities || [];

  if (tmpl.entities.some((e) => e.id === id)) {
    throw new Error(`Entity ID already exists: ${id}`);
  }

  const entity = {
    id,
    enabled,
    instructions,
  };
  if (positiveExample) {
    entity.examples = { positive: [positiveExample], negative: [] };
  }
  tmpl.entities.push(entity);

  if (addToPostpass) {
    tmpl.postpass_alias = tmpl.postpass_alias || {};
    tmpl.postpass_alias.enabled = true;
    tmpl.postpass_alias.entity_ids = Array.from(
      new Set([...(tmpl.postpass_alias.entity_ids || []), id])
    );
  }

  if (provider) {
    tmpl.replacement = tmpl.replacement || {};
    tmpl.replacement.pseudonym = tmpl.replacement.pseudonym || {};
    tmpl.replacement.pseudonym.providers = tmpl.replacement.pseudonym.providers || {};
    tmpl.replacement.pseudonym.providers[id] = provider;
  }

  $("template-json").value = JSON.stringify(tmpl, null, 2);
  loadRuleControls();
  setStatus("template-status", `Added entity ${id}.`, true);

  $("quick-entity-id").value = "";
  $("quick-entity-instructions").value = "";
  $("quick-entity-example").value = "";
  $("quick-entity-provider").value = "";
  $("quick-entity-enable").checked = true;
  $("quick-entity-add-postpass").checked = true;
}

async function loadTemplateIntoEditor() {
  const id = $("editor-template-select").value;
  const data = await api(`/v2/templates/${id}`);
  state.currentTemplate = data;
  $("template-json").value = JSON.stringify(data, null, 2);
  setStatus("template-status", `Loaded ${id}`, true);
  loadRuleControls();
}

async function validateEditorTemplate() {
  const tmpl = parseEditorTemplate();
  const res = await api("/v2/templates/validate", {
    method: "POST",
    body: JSON.stringify(tmpl),
  });
  setStatus(
    "template-status",
    res.valid ? "Template is valid." : `Validation errors:\n${(res.errors || []).join("\n")}`,
    res.valid
  );
}

async function saveEditorTemplate() {
  const tmpl = parseEditorTemplate();
  const customId = $("new-template-id").value.trim();
  if (customId) {
    tmpl.template_id = customId;
  }
  const res = await api("/v2/templates/save", {
    method: "POST",
    body: JSON.stringify(tmpl),
  });
  setStatus("template-status", `Saved ${res.template_id} v${res.version}`, true);
  await loadTemplates();
}

async function deleteEditorTemplate() {
  const tmpl = parseEditorTemplate();
  await api(`/v2/templates/${tmpl.template_id}`, { method: "DELETE" });
  setStatus("template-status", `Deleted ${tmpl.template_id}`, true);
  await loadTemplates();
}

function loadRuleControls() {
  let tmpl;
  try {
    tmpl = parseEditorTemplate();
  } catch {
    return;
  }

  const post = tmpl.postpass_alias || {};
  $("rule-min-token").value = post.min_token_len || 2;
  $("rule-window").value = post.window_size || 2;
  $("rule-overlap").value = post.min_overlap_tokens || 2;

  const wrap = $("entity-toggles");
  wrap.innerHTML = "";
  (tmpl.entities || []).forEach((e) => {
    const label = document.createElement("label");
    label.className = "chip";
    label.innerHTML = `<input type="checkbox" data-entity="${e.id}" ${e.enabled ? "checked" : ""}> ${e.id}`;
    wrap.append(label);
  });
}

function applyRuleControlsToJson() {
  const tmpl = parseEditorTemplate();
  if (!tmpl.postpass_alias) tmpl.postpass_alias = {};
  tmpl.postpass_alias.enabled = true;
  tmpl.postpass_alias.min_token_len = Number($("rule-min-token").value || 2);
  tmpl.postpass_alias.window_size = Number($("rule-window").value || 2);
  tmpl.postpass_alias.min_overlap_tokens = Number($("rule-overlap").value || 2);

  const checked = Array.from(document.querySelectorAll("#entity-toggles input[data-entity]:checked"))
    .map((el) => el.dataset.entity);
  tmpl.postpass_alias.entity_ids = checked;
  tmpl.entities = (tmpl.entities || []).map((e) => ({ ...e, enabled: checked.includes(e.id) }));

  $("template-json").value = JSON.stringify(tmpl, null, 2);
  setStatus("rules-status", "Applied controls to JSON editor.", true);
}

async function validateRules() {
  const tmpl = parseEditorTemplate();
  const res = await api("/v2/templates/validate", {
    method: "POST",
    body: JSON.stringify(tmpl),
  });
  setStatus(
    "rules-status",
    res.valid ? "Rule config is valid." : `Errors:\n${(res.errors || []).join("\n")}`,
    res.valid
  );
}

function renderMapRows(providers = {}) {
  const tbody = $("map-table").querySelector("tbody");
  tbody.innerHTML = "";
  Object.entries(providers).forEach(([entity, provider]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><input value="${entity}" class="map-entity" /></td>
      <td><input value="${provider}" class="map-provider" /></td>
      <td><button class="btn danger remove-row">x</button></td>
    `;
    tbody.append(tr);
  });
}

function collectMapRows() {
  const rows = Array.from($("map-table").querySelectorAll("tbody tr"));
  const providers = {};
  rows.forEach((row) => {
    const entity = row.querySelector(".map-entity").value.trim();
    const provider = row.querySelector(".map-provider").value.trim();
    if (entity && provider) {
      providers[entity] = provider;
    }
  });
  return providers;
}

async function loadMapFromTemplate() {
  const id = $("map-template-select").value;
  const tmpl = await api(`/v2/templates/${id}`);
  state.currentTemplate = tmpl;
  const providers = tmpl?.replacement?.pseudonym?.providers || {};
  renderMapRows(providers);
  setStatus("map-status", `Loaded provider map from ${id}`, true);
}

async function saveMapTemplate() {
  const id = $("map-template-select").value;
  const tmpl = await api(`/v2/templates/${id}`);
  tmpl.replacement = tmpl.replacement || {};
  tmpl.replacement.pseudonym = tmpl.replacement.pseudonym || {};
  tmpl.replacement.pseudonym.providers = collectMapRows();

  await api("/v2/templates/save", {
    method: "POST",
    body: JSON.stringify(tmpl),
  });
  setStatus("map-status", `Saved provider map to ${tmpl.template_id}`, true);
  await loadTemplates();
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });

  $("fake-provider").addEventListener("change", refreshFakeProviderHint);

  $("run-stream").addEventListener("click", async () => {
    try {
      await streamShowcase();
    } catch (e) {
      logTimeline(`error: ${e.message}`);
    }
  });
  $("stop-stream").addEventListener("click", () => {
    if (state.controller) state.controller.abort();
    logTimeline("stream aborted by user");
  });

  $("use-realistic-for-deanon").addEventListener("click", () => {
    $("deanon-input").value = state.realisticText;
  });
  $("use-structured-for-deanon").addEventListener("click", () => {
    $("deanon-input").value = state.structuredText;
  });
  $("run-deanon").addEventListener("click", async () => {
    try {
      await runDeanonymize();
    } catch (e) {
      $("deanon-output").value = `Error: ${e.message}`;
    }
  });

  $("load-template").addEventListener("click", loadTemplateIntoEditor);
  $("add-entity-quick").addEventListener("click", () => {
    try {
      addEntityFromQuickForm();
    } catch (e) {
      setStatus("template-status", e.message, false);
    }
  });
  $("validate-template").addEventListener("click", async () => {
    try {
      await validateEditorTemplate();
    } catch (e) {
      setStatus("template-status", e.message, false);
    }
  });
  $("save-template").addEventListener("click", async () => {
    try {
      await saveEditorTemplate();
    } catch (e) {
      setStatus("template-status", e.message, false);
    }
  });
  $("delete-template").addEventListener("click", async () => {
    try {
      await deleteEditorTemplate();
    } catch (e) {
      setStatus("template-status", e.message, false);
    }
  });

  $("apply-rules").addEventListener("click", () => {
    try {
      applyRuleControlsToJson();
    } catch (e) {
      setStatus("rules-status", e.message, false);
    }
  });
  $("validate-rules").addEventListener("click", async () => {
    try {
      await validateRules();
    } catch (e) {
      setStatus("rules-status", e.message, false);
    }
  });

  $("load-map").addEventListener("click", async () => {
    try {
      await loadMapFromTemplate();
    } catch (e) {
      setStatus("map-status", e.message, false);
    }
  });
  $("add-map-row").addEventListener("click", () => {
    const tbody = $("map-table").querySelector("tbody");
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><input class="map-entity" /></td><td><input class="map-provider" /></td><td><button class="btn danger remove-row">x</button></td>`;
    tbody.append(tr);
  });
  $("map-table").addEventListener("click", (e) => {
    if (e.target.classList.contains("remove-row")) {
      e.target.closest("tr").remove();
    }
  });
  $("save-map").addEventListener("click", async () => {
    try {
      await saveMapTemplate();
    } catch (e) {
      setStatus("map-status", e.message, false);
    }
  });

  $("template-json").addEventListener("input", () => {
    loadRuleControls();
  });
}

async function init() {
  if (!$("health-pill")) return;

  try {
    await api("/health");
    $("health-pill").textContent = "API: healthy";
  } catch {
    $("health-pill").textContent = "API: unavailable";
  }

  await loadTemplates();
  if (state.templates.length) {
    ensureDefaultTemplateSelection();
    await loadTemplateIntoEditor();
    ensureDefaultTemplateSelection();
    await loadMapFromTemplate();
    ensureDefaultTemplateSelection();
  }

  bindEvents();
  refreshFakeProviderHint();
}

init();
