const defaults = {
  budget: 50000000,
  popularity: 12.3,
  runtime: 110,
  vote_average: 7.1,
  vote_count: 3400,
  release_month: 7,
};

function $(id) { return document.getElementById(id); }

let lastResult = null;

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  setTimeout(() => t.classList.add("hidden"), 1800);
}

function readNumber(id) {
  const v = $(id).value;
  const n = Number(v);
  if (Number.isNaN(n)) throw new Error(`Invalid number for ${id}`);
  return n;
}

function buildPayload() {
  return {
    budget: readNumber("budget"),
    popularity: readNumber("popularity"),
    runtime: readNumber("runtime"),
    vote_average: readNumber("vote_average"),
    vote_count: readNumber("vote_count"),
    release_month: readNumber("release_month"),
  };
}

function getHeaders() {
  const headers = { "Content-Type": "application/json" };
  const apiKey = $("api_key")?.value?.trim();
  if (apiKey) headers["X-API-Key"] = apiKey;
  return headers;
}

function baseUrl() {
  const b = $("base_url")?.value?.trim();
  return b ? b.replace(/\/$/, "") : "";
}

function setStatus(kind, label) {
  const b = $("statusBadge");
  b.className = `badge ${kind}`;
  b.textContent = label;
}

function setBusy(isBusy) {
  ["btnAll","btnRevenue","btnClassify","btnCluster","btnReset","btnCopy","btnDownload","btnHealth","btnSample"]
    .filter(Boolean)
    .forEach(id => $(id) && ($(id).disabled = isBusy));
}

function setOutput(obj) {
  $("output").textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
}

function pct(x) {
  if (typeof x !== "number") return String(x);
  return `${(x * 100).toFixed(2)}%`;
}

function card(title, body) {
  const el = document.createElement("div");
  el.className = "resultCard";
  el.innerHTML = `<div class="resultTitle">${escapeHtml(title)}</div>
                  <div class="resultBody">${escapeHtml(body)}</div>`;
  return el;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

function clearCards() {
  const wrap = $("resultCards");
  wrap.innerHTML = "";
  return wrap;
}

function renderRevenue(data) {
  const wrap = clearCards();
  wrap.appendChild(card("Predicted Revenue", data.predicted_revenue_formatted || String(data.predicted_revenue)));
  if (data.confidence_interval) wrap.appendChild(card("Confidence Interval", JSON.stringify(data.confidence_interval)));
  wrap.appendChild(card("Model Used", data.model_used || "unknown"));
}

function renderClassification(data) {
  const wrap = clearCards();
  wrap.appendChild(card("Label", data.prediction_label ?? (data.is_hit ? "Hit" : "Flop")));
  wrap.appendChild(card("Hit Probability", pct(data.hit_probability)));
  wrap.appendChild(card("Flop Probability", pct(data.flop_probability)));
}

function renderCluster(data) {
  const wrap = clearCards();
  wrap.appendChild(card("Cluster ID", String(data.cluster_id)));
  wrap.appendChild(card("Cluster Label", data.cluster_label || `Cluster ${data.cluster_id}`));
  if (data.cluster_profile) wrap.appendChild(card("Cluster Profile", JSON.stringify(data.cluster_profile)));
}

function renderAll(all) {
  const wrap = clearCards();

  if (all.revenue) {
    wrap.appendChild(card("Predicted Revenue", all.revenue.predicted_revenue_formatted || String(all.revenue.predicted_revenue)));
    wrap.appendChild(card("Revenue Model", all.revenue.model_used || "unknown"));
  }

  if (all.classification) {
    wrap.appendChild(card("Hit or Flop", all.classification.prediction_label ?? (all.classification.is_hit ? "Hit" : "Flop")));
    wrap.appendChild(card("Hit Probability", pct(all.classification.hit_probability)));
  }

  if (all.cluster) {
    wrap.appendChild(card("Cluster", `${all.cluster.cluster_label || "Cluster"} (ID ${all.cluster.cluster_id})`));
    if (all.cluster.cluster_profile) wrap.appendChild(card("Cluster Profile", JSON.stringify(all.cluster.cluster_profile)));
  }
}

async function callApi(method, path, payload) {
  const url = baseUrl() + path;
  const t0 = performance.now();

  const opts = { method, headers: getHeaders() };
  if (payload && method !== "GET") opts.body = JSON.stringify(payload);

  const res = await fetch(url, opts);
  const ms = Math.round(performance.now() - t0);
  $("latency").textContent = `latency: ${ms}ms`;

  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { raw: text }; }

  if (!res.ok) {
    const msg = data?.detail || data?.message || JSON.stringify(data);
    throw new Error(`${res.status} ${res.statusText}: ${msg}`);
  }
  return data;
}

async function handle(endpoint) {
  try {
    setBusy(true);
    setStatus("busy", "Running…");
    const payload = buildPayload();
    const data = await callApi("POST", endpoint, payload);

    lastResult = data;

    setStatus("ok", "Success");
    if (endpoint === "/predict/revenue") renderRevenue(data);
    else if (endpoint === "/predict/classification") renderClassification(data);
    else if (endpoint === "/predict/cluster") renderCluster(data);
    else renderAll(data);

    setOutput(data);
  } catch (err) {
    setStatus("err", "Error");
    const obj = { error: String(err) };
    const wrap = clearCards();
    wrap.appendChild(card("Error", obj.error));
    lastResult = obj;
    setOutput(obj);
  } finally {
    setBusy(false);
  }
}

async function predictAll() {
  try {
    setBusy(true);
    setStatus("busy", "Running…");
    const payload = buildPayload();

    const [revenue, classification, cluster] = await Promise.all([
      callApi("POST", "/predict/revenue", payload),
      callApi("POST", "/predict/classification", payload),
      callApi("POST", "/predict/cluster", payload),
    ]);

    const all = { revenue, classification, cluster };
    lastResult = all;

    setStatus("ok", "Success");
    renderAll(all);
    setOutput(all);
  } catch (err) {
    setStatus("err", "Error");
    const obj = { error: String(err) };
    const wrap = clearCards();
    wrap.appendChild(card("Error", obj.error));
    lastResult = obj;
    setOutput(obj);
  } finally {
    setBusy(false);
  }
}

function resetDefaults() {
  Object.entries(defaults).forEach(([k,v]) => $(k).value = v);
  if ($("api_key")) $("api_key").value = "";
  if ($("base_url")) $("base_url").value = "";
  $("latency").textContent = "";
  setStatus("idle", "Idle");

  const wrap = clearCards();
  const tip = document.createElement("div");
  tip.className = "resultCard mutedCard";
  tip.innerHTML = `<div class="resultTitle">Tip</div>
                   <div class="resultBody">Press Predict All for a complete output in one click.</div>`;
  wrap.appendChild(tip);

  setOutput("Run a prediction to see results…");
  lastResult = null;
  toast("Reset done");
}

function fillSample() {
  $("budget").value = 150000000;
  $("popularity").value = 65.4;
  $("runtime").value = 160;
  $("vote_average").value = 7.8;
  $("vote_count").value = 12000;
  $("release_month").value = 12;
  toast("Sample filled");
}

async function copyOutput() {
  try {
    await navigator.clipboard.writeText($("output").textContent);
    toast("Copied JSON");
  } catch {
    toast("Copy failed");
  }
}

function downloadJSON() {
  if (!lastResult) return toast("Nothing to download yet");
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "movie_intelligence_result.json";
  a.click();
  URL.revokeObjectURL(url);
  toast("Downloaded JSON");
}

async function checkHealth() {
  try {
    const data = await callApi("GET", "/health");
    const pill = $("healthPill");
    pill.textContent = `Health: ${data.status || "ok"}`;
    pill.classList.remove("pill-muted");
    toast("Health OK");
  } catch (e) {
    const pill = $("healthPill");
    pill.textContent = "Health: error";
    toast("Health check failed");
  }
}

function setupSegments() {
  const buttons = Array.from(document.querySelectorAll(".segBtn"));
  buttons.forEach(b => {
    b.addEventListener("click", () => {
      buttons.forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      const mode = b.dataset.mode;

      // light UX: highlight relevant buttons
      $("btnRevenue")?.classList.toggle("primary", mode === "revenue");
      $("btnClassify")?.classList.toggle("primary", mode === "classify");
      $("btnCluster")?.classList.toggle("primary", mode === "cluster");
      $("btnAll")?.classList.toggle("primary", mode === "all");
    });
  });
}

// Wiring
$("btnAll").addEventListener("click", predictAll);
$("btnRevenue").addEventListener("click", () => handle("/predict/revenue"));
$("btnClassify").addEventListener("click", () => handle("/predict/classification"));
$("btnCluster").addEventListener("click", () => handle("/predict/cluster"));
$("btnReset").addEventListener("click", resetDefaults);
$("btnSample").addEventListener("click", fillSample);
$("btnCopy").addEventListener("click", copyOutput);
$("btnDownload").addEventListener("click", downloadJSON);
$("btnHealth").addEventListener("click", checkHealth);

setupSegments();
resetDefaults();
checkHealth();
