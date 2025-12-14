function readNumber(id) {
  const v = document.getElementById(id).value;
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
  const apiKey = document.getElementById("api_key").value?.trim();
  if (apiKey) headers["X-API-Key"] = apiKey;
  return headers;
}

async function callApi(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify(payload),
  });

  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { raw: text }; }

  if (!res.ok) {
    const msg = data?.detail || data?.message || JSON.stringify(data);
    throw new Error(`${res.status} ${res.statusText}: ${msg}`);
  }
  return data;
}

function setOutput(obj) {
  document.getElementById("output").textContent = JSON.stringify(obj, null, 2);
}

function setBusy(isBusy) {
  ["btnRevenue", "btnClassify", "btnCluster"].forEach(id => {
    document.getElementById(id).disabled = isBusy;
  });
}

async function handle(path) {
  try {
    setBusy(true);
    setOutput({ status: "running", endpoint: path });
    const payload = buildPayload();
    const data = await callApi(path, payload);
    setOutput(data);
  } catch (err) {
    setOutput({ error: String(err) });
  } finally {
    setBusy(false);
  }
}

document.getElementById("btnRevenue").addEventListener("click", () => handle("/predict/revenue"));
document.getElementById("btnClassify").addEventListener("click", () => handle("/predict/classification"));
document.getElementById("btnCluster").addEventListener("click", () => handle("/predict/cluster"));
