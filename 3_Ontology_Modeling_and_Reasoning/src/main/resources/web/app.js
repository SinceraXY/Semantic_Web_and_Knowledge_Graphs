async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || ("HTTP " + res.status));
  }
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data && data.error ? data.error : ("HTTP " + res.status);
    throw new Error(msg);
  }
  return data;
}

function qs(sel) {
  return document.querySelector(sel);
}

let lastResult = null;
let toastTimer = null;

function toast(message) {
  const el = qs("#toast");
  if (!el) return;
  el.textContent = message;
  el.classList.add("show");
  if (toastTimer) {
    clearTimeout(toastTimer);
  }
  toastTimer = setTimeout(() => {
    el.classList.remove("show");
  }, 1800);
}

async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("已复制到剪贴板");
  } catch (e) {
    toast("复制失败（浏览器权限限制）");
  }
}

function getModel() {
  const el = document.querySelector("input[name='model']:checked");
  return el ? el.value : "base";
}

function setMeta(text) {
  qs("#meta").textContent = text;
}

function setResultMeta(text) {
  qs("#resultMeta").textContent = text;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderTable(vars, rows) {
  const header = "<tr>" + vars.map(v => `<th>${escapeHtml(v)}</th>`).join("") + "</tr>";
  const body = rows.map(r => {
    const tds = vars.map(v => `<td>${escapeHtml(r[v] ?? "")}</td>`).join("");
    return `<tr>${tds}</tr>`;
  }).join("");

  return `<table class='table'><thead>${header}</thead><tbody>${body}</tbody></table>`;
}

function setResultHtml(html) {
  qs("#result").innerHTML = html;
}

function resultToTsv(result) {
  if (!result) return "";
  if (result.booleanResult !== undefined && result.booleanResult !== null) {
    return String(result.booleanResult);
  }
  const vars = result.vars || [];
  const rows = result.rows || [];
  const lines = [];
  lines.push(vars.join("\t"));
  for (const r of rows) {
    const line = vars.map(v => (r[v] ?? "").replaceAll("\t", " ").replaceAll("\n", " ")).join("\t");
    lines.push(line);
  }
  return lines.join("\n");
}

function selectModel(value) {
  const el = document.querySelector(`input[name='model'][value='${value}']`);
  if (el) {
    el.checked = true;
  }
}

function setInstances(list) {
  const box = qs("#instancesList");
  if (!list || list.length === 0) {
    box.innerHTML = "<div class='empty'>无实例</div>";
    return;
  }
  box.innerHTML = list.map(x => `<div class='chip'>${escapeHtml(x)}</div>`).join("");
}

function setProps(list) {
  const box = qs("#propsList");
  if (!list || list.length === 0) {
    box.innerHTML = "<div class='empty'>无属性</div>";
    return;
  }
  box.innerHTML = list.map(x => `<div class='chip'>${escapeHtml(x)}</div>`).join("");
}

const examples = {
  students: "SELECT ?s WHERE { ?s rdf:type :Student . } ORDER BY ?s",
  studentCourses: "SELECT ?student ?course WHERE { ?student a :Student . ?student :enrolledIn ?course . } ORDER BY ?student ?course",
  classHierarchy: "PREFIX owl: <http://www.w3.org/2002/07/owl#>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nSELECT ?class ?super WHERE { ?class a owl:Class . FILTER(STRSTARTS(STR(?class), STR(:))) OPTIONAL { ?class rdfs:subClassOf ?super . } } ORDER BY ?class ?super",
  propDomainRange: "PREFIX owl: <http://www.w3.org/2002/07/owl#>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nSELECT ?prop ?type ?domain ?range WHERE { ?prop a ?type . FILTER(?type IN (owl:ObjectProperty, owl:DatatypeProperty)) FILTER(STRSTARTS(STR(?prop), STR(:))) OPTIONAL { ?prop rdfs:domain ?domain . } OPTIONAL { ?prop rdfs:range ?range . } } ORDER BY ?prop",
  taughtBy: "SELECT ?student ?teacher WHERE { ?student :isTaughtBy ?teacher . } ORDER BY ?student ?teacher",
  excellent: "SELECT ?student ?flag WHERE { ?student a :ExcellentStudent . OPTIONAL { ?student :eligibleForScholarship ?flag . } } ORDER BY ?student"
};

async function refreshSummary() {
  const summary = await apiGet("/api/summary");
  setMeta(
    `base triples: ${summary.baseTriples} | inferred triples: ${summary.inferredTriples}\n` +
    `ns: ${summary.ns}\n` +
    `ontology: ${summary.ontology || ""}\n` +
    `instances: ${summary.instances || ""}\n` +
    `rules: ${summary.rules || ""}`
  );
}

async function loadClasses() {
  const m = getModel();
  const classes = await apiGet(`/api/ontology/classes?model=${encodeURIComponent(m)}`);
  const sel = qs("#classSelect");
  sel.innerHTML = "";
  for (const c of classes) {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    sel.appendChild(opt);
  }
}

async function loadProps() {
  const m = getModel();
  const props = await apiGet(`/api/ontology/properties?model=${encodeURIComponent(m)}`);
  setProps(props);
}

async function loadInstances() {
  const sel = qs("#classSelect");
  const v = sel.value;
  if (!v) {
    setInstances([]);
    return;
  }
  const m = getModel();
  const inds = await apiGet(`/api/instances?model=${encodeURIComponent(m)}&class=${encodeURIComponent(v)}`);
  setInstances(inds);
}

async function runQuery() {
  const sparql = qs("#sparql").value;
  const model = getModel();

  setResultMeta("执行中...");
  setResultHtml("");
  lastResult = null;

  const t0 = performance.now();
  try {
    const out = await apiPost("/api/query", { sparql, model });
    const dt = Math.round(performance.now() - t0);

    if (out.error) {
      setResultMeta(`错误（${out.elapsedMs ?? dt} ms）`);
      setResultHtml(`<pre class='error'>${escapeHtml(out.error)}</pre>`);
      lastResult = out;
      return;
    }

    if (out.booleanResult !== undefined && out.booleanResult !== null) {
      setResultMeta(`ASK 结果（${out.elapsedMs ?? dt} ms）`);
      setResultHtml(`<div class='ask'>${out.booleanResult ? "true" : "false"}</div>`);
      lastResult = out;
      return;
    }

    const vars = out.vars || [];
    const rows = out.rows || [];
    setResultMeta(`SELECT 结果：${rows.length} 行（${out.elapsedMs ?? dt} ms）`);
    if (rows.length === 0) {
      setResultHtml("<div class='empty'>无结果。你可以尝试切换 base/inf 或检查前缀/类名。</div>");
    } else {
      setResultHtml(renderTable(vars, rows));
    }
    lastResult = out;
  } catch (e) {
    setResultMeta("执行失败");
    setResultHtml(`<pre class='error'>${escapeHtml(e.message || String(e))}</pre>`);
    lastResult = { error: e.message || String(e) };
  }
}

function bindExamples() {
  document.querySelectorAll("button[data-example]").forEach(btn => {
    btn.addEventListener("click", () => {
      const key = btn.getAttribute("data-example");
      qs("#sparql").value = examples[key] || "";
      if (key === "taughtBy" || key === "excellent") {
        selectModel("inf");
        refreshSummary();
        toast("已切换到 inf（含推理结果）");
      }
    });
  });
}

function bindEvents() {
  qs("#btnLoadClasses").addEventListener("click", async () => {
    await loadClasses();
  });

  qs("#btnLoadProps").addEventListener("click", async () => {
    await loadProps();
  });

  qs("#btnLoadInstances").addEventListener("click", async () => {
    await loadInstances();
  });

  qs("#btnRun").addEventListener("click", async () => {
    await runQuery();
  });

  qs("#sparql").addEventListener("keydown", async (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      await runQuery();
    }
  });

  qs("#btnCopyQuery").addEventListener("click", async () => {
    await copyToClipboard(qs("#sparql").value || "");
  });

  qs("#btnCopyResult").addEventListener("click", async () => {
    const tsv = resultToTsv(lastResult);
    if (!tsv) {
      toast("暂无可复制结果");
      return;
    }
    await copyToClipboard(tsv);
  });

  qs("#btnClear").addEventListener("click", () => {
    qs("#sparql").value = "";
    setResultMeta("");
    setResultHtml("");
  });

  document.querySelectorAll("input[name='model']").forEach(r => {
    r.addEventListener("change", async () => {
      await refreshSummary();
      // 不自动重载 classes/props，避免意外刷新用户选择
    });
  });
}

async function main() {
  bindExamples();
  bindEvents();
  qs("#sparql").value = examples.students;
  await refreshSummary();
  toast("页面已就绪");
}

main().catch(err => {
  setResultMeta("页面初始化失败");
  setResultHtml(`<pre class='error'>${escapeHtml(err.message || String(err))}</pre>`);
});
