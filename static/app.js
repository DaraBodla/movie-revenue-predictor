/*  src/ui/static/app.js  (FIXED)
    - Fixes UI path issues for TMDB + prediction endpoints
    - Uses SAME baseUrl() for GET and POST (works with Docker/hosted)
    - Robust number parsing (empty => error message)
    - Fixes classification endpoint mismatch (tries /predict/hitflop then falls back to /predict/classification)
    - Fixes genres typo in defaults ("Adventure")
    - Autofill also fills genres if returned by backend
*/

const defaults = {
  budget: 50000000,
  popularity: 12.3,
  runtime: 110,
  vote_average: 7.1,
  vote_count: 3400,
  release_month: 7,
  genres: "Action, Adventure",
};

function $(id) { return document.getElementById(id); }

function toast(msg) {
  const t = $("toast");
  if (!t) return;
  t.textContent = msg;
  t.classList.remove("hidden");
  setTimeout(() => t.classList.add("hidden"), 1800);
}

function readNumber(id) {
  const el = $(id);
  if (!el) throw new Error(Missing input element: ${id});
  const raw = String(el.value ?? "").trim();
  if (raw === "") throw new Error(Missing value for ${id});
  const n = Number(raw);
  if (!Number.isFinite(n)) throw new Error(Invalid number for ${id});
  return n;
}

function buildPayload() {
  const genresEl = $("genres");
  const genres = genresEl ? String(genresEl.value ?? "").trim() : "";
  return {
    budget: readNumber("budget"),
    popularity: readNumber("popularity"),
    runtime: readNumber("runtime"),
    vote_average: readNumber("vote_average"),
    vote_count: readNumber("vote_count"),
    release_month: readNumber("release_month"),
    // optional
    genres: genres || null,
  };
}

function getHeaders() {
  const headers = { "Content-Type": "application/json" };
  const apiKeyEl = $("api_key");
  const apiKey = apiKeyEl ? apiKeyEl.value?.trim() : "";
  if (apiKey) headers["X-API-Key"] = apiKey;
  return headers;
}

function baseUrl() {
  const el = $("base_url");
  const b = el ? String(el.value ?? "").trim() : "";
  return b ? b.replace(/\/$/, "") : "";
}

function setLatency(ms) {
  const el = $("latency");
  if (el) el.textContent = latency: ${ms}ms;
}

async function parseJsonOrRaw(res) {
  const text = await res.text();
  try { return text ? JSON.parse(text) : null; } catch { return { raw: text }; }
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
  setLatency(ms);

  const data = await parseJsonOrRaw(res);

  if (!res.ok) {
    const msg = data?.detail || data?.message || JSON.stringify(data);
    throw new Error(${res.status} ${res.statusText}: ${msg});
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
  setLatency(ms);

  const data = await parseJsonOrRaw(res);

  if (!res.ok) {
    const msg = data?.detail || data?.message || JSON.stringify(data);
    throw new Error(${res.status} ${res.statusText}: ${msg});
  }
  return data;
}

function setStatus(kind, label) {
  const b = $("statusBadge");
  if (!b) return;
  b.className = badge ${kind};
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
  const out = $("output");
  if (!out) return;
  out.textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
}

function pct(x) {
  if (typeof x !== "number") return String(x);
  return ${(x * 100).toFixed(2)}%;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

function card(title, body) {
  const el = document.createElement("div");
  el.className = "resultCard";
  el.innerHTML = `<div class="resultTitle">${escapeHtml(title)}</div>
                  <div class="resultBody">${escapeHtml(body)}</div>`;
  return el;
}

function renderCards(endpoint, data) {
  const wrap = $("resultCards");
  if (!wrap) return;

  const cards = [];

  if (endpoint === "/predict/revenue") {
    cards.push(card("Revenue", data.predicted_revenue_formatted || String(data.predicted_revenue)));
    if (data.confidence_interval) cards.push(card("Confidence Interval", JSON.stringify(data.confidence_interval)));
    cards.push(card("Model", data.model_used || "unknown"));

  } else if (endpoint === "/predict/hitflop" || endpoint === "/predict/classification") {
    // support both response shapes
    const label =
      data.prediction_label ??
      (typeof data.is_hit === "boolean" ? (data.is_hit ? "Hit" : "Flop") : undefined) ??
      data.label ??
      "Unknown";
    cards.push(card("Label", String(label)));

    if (data.hit_probability !== undefined) cards.push(card("Hit Probability", pct(data.hit_probability)));
    if (data.flop_probability !== undefined) cards.push(card("Flop Probability", pct(data.flop_probability)));
    if (data.probability !== undefined) cards.push(card("Probability", pct(data.probability)));

  } else if (endpoint === "/predict/cluster") {
    cards.push(card("Cluster ID", String(data.cluster_id)));
    cards.push(card("Cluster Label", data.cluster_label || Cluster ${data.cluster_id}));
    if (data.cluster_profile) cards.push(card("Profile", JSON.stringify(data.cluster_profile)));

  } else {
    cards.push(card("Result", JSON.stringify(data)));
  }

  wrap.innerHTML = "";
  cards.forEach(c => wrap.appendChild(c));
}

async function handleRevenue() {
  await handle("/predict/revenue");
}

async function handleCluster() {
  await handle("/predict/cluster");
}

/**
 * IMPORTANT FIX:
 * Your backend in earlier logs used /predict/hitflop.
 * Your UI had /predict/classification.
 * We'll try /predict/hitflop first, then fallback to /predict/classification if 404.
 */
async function handleClassify() {
  try {
    await handle("/predict/hitflop");
  } catch (e) {
    // Only fallback if it looks like a 404
    if (String(e).includes("404")) {
      await handle("/predict/classification");
    } else {
      throw e;
    }
  }
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
    const wrap = $("resultCards");
    if (wrap) {
      wrap.innerHTML = "";
      wrap.appendChild(card("Error", obj.error));
    }
    setOutput(obj);
  } finally {
    setBusy(false);
  }
}

function resetDefaults() {
  Object.entries(defaults).forEach(([k, v]) => {
    const el = $(k);
    if (el) el.value = v;
  });

  const apiKeyEl = $("api_key");
  if (apiKeyEl) apiKeyEl.value = "";

  setLatency("");
  const lat = $("latency");
  if (lat) lat.textContent = "";

  const q = $("tmdb_query");
  if (q) q.value = "";
  const y = $("tmdb_year");
  if (y) y.value = "";

  const sel = $("tmdb_results");
  if (sel) sel.innerHTML = <option value="">Search to load results…</option>;

  const fillBtn = $("btnTmdbFill");
  if (fillBtn) fillBtn.disabled = true;

  setStatus("idle", "Idle");

  const wrap = $("resultCards");
  if (wrap) {
    wrap.innerHTML = "";
    wrap.appendChild(card("Tip", "Run a prediction to see structured results here."));
  }

  setOutput("Run a prediction to see results…");
  toast("Reset done");
}

async function copyOutput() {
  try {
    const out = $("output");
    if (!out) return;
    await navigator.clipboard.writeText(out.textContent || "");
    toast("Copied JSON");
  } catch {
    toast("Copy failed");
  }
}

function setupSegments() {
  const buttons = Array.from(document.querySelectorAll(".segBtn"));
  buttons.forEach((b) => {
    b.addEventListener("click", () => {
      buttons.forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      const mode = b.dataset.mode;

      const rev = $("btnRevenue");
      const cls = $("btnClassify");
      const clu = $("btnCluster");

      if (rev) rev.classList.toggle("primary", mode === "revenue" || mode === "all");
      if (cls) cls.classList.toggle("primary", mode === "classify");
      if (clu) clu.classList.toggle("primary", mode === "cluster");
    });
  });
}

/* =========================
   TMDB LIVE AUTOFILL (FIXED PATH)
   Backend must expose:
   GET /live/tmdb/search?q=...&year=...
   GET /live/tmdb/{id}/inputs
   ========================= */

let tmdbCache = [];

function formatTmdbOption(r) {
  const title = r.title || "Untitled";
  const date = r.release_date ? (${r.release_date}) : "(no date)";
  const votes = (typeof r.vote_count === "number") ? ` • votes: ${r.vote_count}` : "";
  return ${title} ${date}${votes};
}

async function tmdbSearch() {
  const qEl = $("tmdb_query");
  const yEl = $("tmdb_year");
  const q = qEl ? qEl.value.trim() : "";
  const yearRaw = yEl ? yEl.value.trim() : "";
  const year = yearRaw ? Number(yearRaw) : null;

  if (!q) {
    toast("Enter a movie title");
    return;
  }

  try {
    setBusy(true);
    setStatus("busy", "Searching TMDB…");

    const qs = new URLSearchParams({ q });
    if (year && Number.isFinite(year)) qs.set("year", String(year));

    // FIX: This must match backend route
    const data = await callGet(/live/tmdb/search?${qs.toString()});

    tmdbCache = Array.isArray(data) ? data : [];

    const sel = $("tmdb_results");
    if (!sel) return;

    sel.innerHTML = "";

    if (tmdbCache.length === 0) {
      sel.innerHTML = <option value="">No results found</option>;
      const fillBtn = $("btnTmdbFill");
      if (fillBtn) fillBtn.disabled = true;
      setStatus("idle", "Idle");
      return;
    }

    sel.appendChild(new Option("Select a result…", ""));
    tmdbCache.forEach((r) => {
      sel.appendChild(new Option(formatTmdbOption(r), String(r.id)));
    });

    const fillBtn = $("btnTmdbFill");
    if (fillBtn) fillBtn.disabled = true;

    setStatus("ok", Found ${tmdbCache.length});
    toast(Found ${tmdbCache.length} results);
  } catch (err) {
    setStatus("err", "TMDB error");
    setOutput({ error: String(err) });
    toast("TMDB search failed");
  } finally {
    setBusy(false);
  }
}

async function tmdbAutofill() {
  const sel = $("tmdb_results");
  const id = sel ? sel.value : "";
  if (!id) {
    toast("Select a TMDB result first");
    return;
  }

  try {
    setBusy(true);
    setStatus("busy", "Loading TMDB details…");

    // FIX: This must match backend route
    const inputs = await callGet(/live/tmdb/${encodeURIComponent(id)}/inputs);

    // Fill best effort
    const fields = ["budget","popularity","runtime","vote_average","vote_count","release_month"];
    fields.forEach((f) => {
      const el = $(f);
      if (!el) return;
      if (inputs && inputs[f] !== undefined && inputs[f] !== null) {
        el.value = inputs[f];
      }
    });

    // ALSO fill genres (Option A)
    const genresEl = $("genres");
    if (genresEl && inputs && inputs.genres) {
      genresEl.value = inputs.genres;
    }

    setStatus("ok", "Autofilled");
    setOutput({ message: "Autofilled inputs from TMDB", tmdb: inputs });
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
  const sel = $("tmdb_results");
  const fillBtn = $("btnTmdbFill");
  if (!fillBtn) return;
  const hasSelection = Boolean(sel && sel.value);
  fillBtn.disabled = !hasSelection;
}

/* =========================
   Event bindings
   ========================= */

const btnRevenue = $("btnRevenue");
if (btnRevenue) btnRevenue.addEventListener("click", handleRevenue);

const btnClassify = $("btnClassify");
if (btnClassify) btnClassify.addEventListener("click", handleClassify);

const btnCluster = $("btnCluster");
if (btnCluster) btnCluster.addEventListener("click", handleCluster);

const btnReset = $("btnReset");
if (btnReset) btnReset.addEventListener("click", resetDefaults);

const btnCopy = $("btnCopy");
if (btnCopy) btnCopy.addEventListener("click", copyOutput);

// TMDB hooks
const btnTmdbSearch = $("btnTmdbSearch");
if (btnTmdbSearch) btnTmdbSearch.addEventListener("click", tmdbSearch);

const btnTmdbFill = $("btnTmdbFill");
if (btnTmdbFill) btnTmdbFill.addEventListener("click", tmdbAutofill);

const tmdbResults = $("tmdb_results");
if (tmdbResults) tmdbResults.addEventListener("change", onTmdbSelectionChange);

const tmdbQuery = $("tmdb_query");
if (tmdbQuery) {
  tmdbQuery.addEventListener("keydown", (e) => {
    if (e.key === "Enter") tmdbSearch();
  });
}

setupSegments();
resetDefaults();
