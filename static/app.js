const defaults = {
  budget: 50000000,
  popularity: 12.3,
  runtime: 110,
  vote_average: 7.1,
  vote_count: 3400,
  release_month: 7,
  genres: "Action, Adventue",
};

function $(id) { return document.getElementById(id); }

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
    genres: document.getElementById("genres").value.trim() || null,
  };
}

function getHeaders() {
  const headers = { "Content-Type": "application/json" };
  const apiKey = $("api_key").value?.trim();
  if (apiKey) headers["X-API-Key"] = apiKey;
  return headers;
}

function baseUrl() {
  const b = $("base_url").value.trim();
  return b ? b.replace(/\/$/, "") : "";
}

async function callApi(path, payload) {
  const url = baseUrl() + path;
  const t0 = performance.now();

  const res = await fetch(url, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify(payload),
  });

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

async function callGet(path) {
  const url = baseUrl() + path;
  const t0 = performance.now();

  const res = await fetch(url, {
    method: "GET",
    headers: getHeaders(),
  });

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

function setStatus(kind, label) {
  const b = $("statusBadge");
  b.className = `badge ${kind}`;
  b.textContent = label;
}

function setBusy(isBusy) {
  [
    "btnRevenue","btnClassify","btnCluster","btnReset","btnCopy",
    "btnTmdbSearch","btnTmdbFill"
  ].forEach(id => {
    const el = $(id);
    if (el) el.disabled = isBusy;
  });
}

function setOutput(obj) {
  $("output").textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
}

function renderCards(endpoint, data) {
  const wrap = $("resultCards");
  const cards = [];

  if (endpoint === "/predict/revenue") {
    cards.push(card("Revenue", data.predicted_revenue_formatted || String(data.predicted_revenue)));
    if (data.confidence_interval) {
      cards.push(card("Confidence Interval", JSON.stringify(data.confidence_interval)));
    }
    cards.push(card("Model", data.model_used || "unknown"));
  } else if (endpoint === "/predict/classification") {
    cards.push(card("Label", data.prediction_label ?? (data.is_hit ? "Hit" : "Flop")));
    cards.push(card("Hit Probability", pct(data.hit_probability)));
    cards.push(card("Flop Probability", pct(data.flop_probability)));
  } else if (endpoint === "/predict/cluster") {
    cards.push(card("Cluster ID", String(data.cluster_id)));
    cards.push(card("Cluster Label", data.cluster_label || `Cluster ${data.cluster_id}`));
    if (data.cluster_profile) cards.push(card("Profile", JSON.stringify(data.cluster_profile)));
  } else {
    cards.push(card("Result", JSON.stringify(data)));
  }

  wrap.innerHTML = "";
  cards.forEach(c => wrap.appendChild(c));
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

async function handle(endpoint) {
  try {
    setBusy(true);
    setStatus("busy", "Running…");
    const payload = buildPayload();
    const data = await callApi(endpoint, payload);
    setStatus("ok", "Success");
    renderCards(endpoint, data);
    setOutput(data);
  } catch (err) {
    setStatus("err", "Error");
    const obj = { error: String(err) };
    $("resultCards").innerHTML = "";
    $("resultCards").appendChild(card("Error", obj.error));
    setOutput(obj);
  } finally {
    setBusy(false);
  }
}

function resetDefaults() {
  Object.entries(defaults).forEach(([k,v]) => $(k).value = v);
  $("api_key").value = "";
  $("latency").textContent = "";
  $("tmdb_query").value = "";
  $("tmdb_year").value = "";
  $("tmdb_results").innerHTML = `<option value="">Search to load results…</option>`;
  $("btnTmdbFill").disabled = true;

  setStatus("idle", "Idle");
  $("resultCards").innerHTML = "";
  $("resultCards").appendChild(card("Tip", "Run a prediction to see structured results here."));
  setOutput("Run a prediction to see results…");
  toast("Reset done");
}

async function copyOutput() {
  try {
    await navigator.clipboard.writeText($("output").textContent);
    toast("Copied JSON");
  } catch {
    toast("Copy failed");
  }
}

function setupSegments() {
  const buttons = Array.from(document.querySelectorAll(".segBtn"));
  buttons.forEach(b => {
    b.addEventListener("click", () => {
      buttons.forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      const mode = b.dataset.mode;

      $("btnRevenue").classList.toggle("primary", mode === "revenue" || mode === "all");
      $("btnClassify").classList.toggle("primary", mode === "classify");
      $("btnCluster").classList.toggle("primary", mode === "cluster");
    });
  });
}

/* =========================
   TMDB LIVE AUTOFILL
   ========================= */

let tmdbCache = []; // store latest search results

function formatTmdbOption(r) {
  const title = r.title || "Untitled";
  const date = r.release_date ? `(${r.release_date})` : "(no date)";
  const votes = (typeof r.vote_count === "number") ? ` • votes: ${r.vote_count}` : "";
  return `${title} ${date}${votes}`;
}

async function tmdbSearch() {
  const q = $("tmdb_query").value.trim();
  const yearRaw = $("tmdb_year").value.trim();
  const year = yearRaw ? Number(yearRaw) : null;

  if (!q) {
    toast("Enter a movie title");
    return;
  }

  try {
    setBusy(true);
    setStatus("busy", "Searching TMDB…");

    const qs = new URLSearchParams({ q });
    if (year && !Number.isNaN(year)) qs.set("year", String(year));

    const data = await callGet(`/live/tmdb/search?${qs.toString()}`);
    tmdbCache = Array.isArray(data) ? data : [];

    const sel = $("tmdb_results");
    sel.innerHTML = "";

    if (tmdbCache.length === 0) {
      sel.innerHTML = `<option value="">No results found</option>`;
      $("btnTmdbFill").disabled = true;
      setStatus("idle", "Idle");
      return;
    }

    sel.appendChild(new Option("Select a result…", ""));
    tmdbCache.forEach((r) => {
      const opt = new Option(formatTmdbOption(r), String(r.id));
      sel.appendChild(opt);
    });

    $("btnTmdbFill").disabled = true;
    setStatus("ok", `Found ${tmdbCache.length}`);
    toast(`Found ${tmdbCache.length} results`);
  } catch (err) {
    setStatus("err", "TMDB error");
    setOutput({ error: String(err) });
    toast("TMDB search failed");
  } finally {
    setBusy(false);
  }
}

async function tmdbAutofill() {
  const id = $("tmdb_results").value;
  if (!id) {
    toast("Select a TMDB result first");
    return;
  }

  try {
    setBusy(true);
    setStatus("busy", "Loading TMDB details…");

    const inputs = await callGet(`/live/tmdb/${encodeURIComponent(id)}/inputs`);

    // Fill ONLY if field exists (best effort)
    const fields = ["budget","popularity","runtime","vote_average","vote_count","release_month"];
    fields.forEach((f) => {
      if (inputs && inputs[f] !== undefined && inputs[f] !== null) {
        $(f).value = inputs[f];
      }
    });

    setStatus("ok", "Autofilled");
    setOutput({
      message: "Autofilled inputs from TMDB",
      tmdb: inputs
    });
    toast("Autofill complete");
  } catch (err) {
    setStatus("err", "Autofill error");
    setOutput({ error: String(err) });
    toast("Autofill failed");
  } finally {
    setBusy(false);
  }
}

function onTmdbSelectionChange() {
  const hasSelection = Boolean($("tmdb_results").value);
  $("btnTmdbFill").disabled = !hasSelection;
}

/* ========================= */

$("btnRevenue").addEventListener("click", () => handle("/predict/revenue"));
$("btnClassify").addEventListener("click", () => handle("/predict/classification"));
$("btnCluster").addEventListener("click", () => handle("/predict/cluster"));
$("btnReset").addEventListener("click", resetDefaults);
$("btnCopy").addEventListener("click", copyOutput);

// TMDB hooks
$("btnTmdbSearch").addEventListener("click", tmdbSearch);
$("btnTmdbFill").addEventListener("click", tmdbAutofill);
$("tmdb_results").addEventListener("change", onTmdbSelectionChange);
$("tmdb_query").addEventListener("keydown", (e) => {
  if (e.key === "Enter") tmdbSearch();
});

setupSegments();
resetDefaults();
