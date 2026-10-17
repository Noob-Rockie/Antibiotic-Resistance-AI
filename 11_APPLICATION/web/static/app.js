/* Antimicrobial Resistance CDS — frontend application logic.
 * No framework: plain DOM rendering against the FastAPI backend in
 * backend.py. Every number displayed comes from that API — this file only
 * renders it.
 */
"use strict";

const SAFETY_DISCLAIMER =
  "This application is a research / clinical decision-support prototype. " +
  "It does not replace qualified healthcare professionals, antimicrobial " +
  "susceptibility testing, clinical guidelines, or professional medical " +
  "judgement. AI-generated predictions and recommendations must be " +
  "reviewed by an appropriately qualified healthcare professional.";

const STATE = {
  view: "dashboard",
  modelStatus: null,
  meta: null,          // /api/specimens + /api/history-fields, cached
  wizard: freshWizard(),
  analysis: null,       // last /api/analyze(-sample) response
  detailCache: { dosage: {}, cost: {}, genomic: null, patientHistory: null,
                groundTruth: null },
  insights: null,
  methodology: null,
};

function freshWizard() {
  return {
    step: 0,
    mode: "search",
    query: "",
    matches: [],
    organism: null,
    sampleOrders: null,
    sampleOrderId: null,
    sampleOrganisms: [],
    sampleOrganism: null,
    ctx: {
      specimen: "URINE", morphotype: "standard",
      order_year: new Date().getUTCFullYear(),
      genderChoice: "Not recorded", setting: "Inpatient",
      ageBand: null, historyUnknown: false, history: {}, labs: {},
      allergiesRaw: "",
    },
  };
}

// ------------------------------------------------------------- utilities
function el(id) { return document.getElementById(id); }
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
function fmtPct(x, digits = 0) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return (x * 100).toFixed(digits) + "%";
}
function fmtNum(x) {
  if (x === null || x === undefined) return "—";
  if (typeof x === "number") return x.toLocaleString();
  return String(x);
}
function loadingHTML(msg = "Loading…") {
  return `<div class="loading-inline"><span class="spinner"></span> ${escapeHtml(msg)}</div>`;
}
function safetyBanner(compact = false) {
  return `<div class="banner${compact ? " compact" : ""}"><strong>${SAFETY_DISCLAIMER}</strong></div>`;
}
function statCard(label, value, sub) {
  return `<div class="stat-card"><div class="label">${escapeHtml(label)}</div>` +
    `<div class="value">${escapeHtml(String(value))}</div>` +
    (sub ? `<div class="sub">${escapeHtml(sub)}</div>` : "") + `</div>`;
}
function metric(label, value) {
  return `<div class="metric"><div class="m-label">${escapeHtml(label)}</div>` +
    `<div class="m-value">${escapeHtml(String(value ?? "—"))}</div></div>`;
}
function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch (_) { /* no body */ }
  if (!res.ok) {
    const detail = body && body.detail ? body.detail : body;
    const err = new Error((detail && detail.message) || res.statusText);
    err.detail = detail;
    err.status = res.status;
    throw err;
  }
  return body;
}
function errorBanner(err) {
  const d = err && err.detail;
  const message = (d && d.message) || err.message || "Something went wrong.";
  const reason = (d && (d.reason || d.recommended_action)) || "";
  return `<div class="banner error"><strong>${escapeHtml(message)}</strong>` +
    (reason ? `<br>${escapeHtml(reason)}` : "") + `</div>`;
}

function riskCategory(p) {
  const v = p == null ? 0 : p;
  if (v < 0.15) return ["LOW RESISTANCE RISK", "risk-low"];
  if (v < 0.30) return ["LOW-MODERATE RESISTANCE RISK", "risk-low-moderate"];
  if (v < 0.50) return ["MODERATE RESISTANCE RISK", "risk-moderate"];
  return ["HIGH RESISTANCE RISK", "risk-high"];
}
function riskBadge(p) {
  const [label, cls] = riskCategory(p);
  return `<span class="risk-badge ${cls}">${escapeHtml(label)} &middot; ${fmtPct(p)}</span>`;
}
function awarePill(cat) {
  const cls = { Access: "access", Watch: "watch", Reserve: "reserve" }[cat] || "neutral";
  return `<span class="pill ${cls}">AWaRe: ${escapeHtml(cat || "—")}</span>`;
}
function evidencePill(strength) {
  let cls = "neutral";
  if ((strength || "").startsWith("strong")) cls = "access";
  else if ((strength || "").startsWith("moderate")) cls = "blue";
  else if ((strength || "").startsWith("none")) cls = "reserve";
  return `<span class="pill ${cls}">evidence: ${escapeHtml(strength || "—")}</span>`;
}
function flagBox(flag) {
  const sev = flag.severity || "info";
  return `<div class="flag-box ${sev}"><strong>${escapeHtml(flag.message || "")}</strong><br>` +
    `${escapeHtml(flag.detail || "")}<span class="flag-source">Source: ${escapeHtml(flag.source || "—")}</span></div>`;
}

// ---------------------------------------------------------------- router
function navigate(view) {
  STATE.view = view;
  document.querySelectorAll("#sidebar .nav-item").forEach((a) => {
    a.classList.toggle("active", a.dataset.view === view);
  });
  renderView();
  window.scrollTo(0, 0);
}

document.querySelectorAll("#sidebar .nav-item").forEach((a) => {
  a.addEventListener("click", () => {
    if (a.dataset.view === "new-analysis") STATE.wizard = freshWizard();
    navigate(a.dataset.view);
  });
});

function renderView() {
  const root = el("view-root");
  root.innerHTML = loadingHTML();
  const fn = {
    "dashboard": renderDashboard,
    "new-analysis": renderNewAnalysis,
    "results": renderResults,
    "model-insights": renderModelInsights,
    "how-it-works": renderHowItWorks,
  }[STATE.view];
  fn().catch((err) => { root.innerHTML = errorBanner(err); });
}

async function refreshModelStatus() {
  try {
    STATE.modelStatus = await api("/api/model-status");
  } catch (err) {
    STATE.modelStatus = { ready: false, reason: String(err.message) };
  }
  const box = el("model-status");
  const text = el("model-status-text");
  if (STATE.modelStatus.ready) {
    box.classList.remove("down");
    text.textContent = "Model System Ready";
  } else {
    box.classList.add("down");
    text.textContent = "Model System Unavailable";
  }
}

// ================================================================ DASHBOARD
async function renderDashboard() {
  if (!STATE.modelStatus) await refreshModelStatus();
  const s = STATE.modelStatus;
  const root = el("view-root");
  root.innerHTML = `
    <div class="eyebrow">Dashboard</div>
    <h1>Antimicrobial Resistance</h1>
    <p class="sub">Clinical Decision Support &middot; AI-Assisted Resistance Prediction</p>
    <button class="btn primary" id="start-analysis" ${s.ready ? "" : "disabled"}>Start New Analysis</button>

    <div class="eyebrow" style="margin-top:2.2rem;">System overview</div>
    <div class="stat-grid">
      ${statCard("ML Model", s.ready ? s.algorithm : "—")}
      ${statCard("Explainability", "SHAP Enabled", "TreeSHAP, per-prediction")}
      ${statCard("Supported Organisms", s.ready ? s.n_organisms : "—", "Dataset-defined vocabulary")}
      ${statCard("Model Status", s.ready ? "Ready" : "Unavailable",
                s.ready ? `${s.n_antibiotics} antimicrobial agents` : (s.reason || ""))}
    </div>

    <div class="eyebrow">How it works</div>
    <div class="card">
      <ol class="workflow-list">
        <li>Select bacterial organism</li>
        <li>Enter required clinical context</li>
        <li>Analyze resistance profile</li>
        <li>Review predicted resistance probabilities</li>
        <li>Review ranked antimicrobial options</li>
        <li>Review AI explanation</li>
        <li>Make clinician-led decision</li>
      </ol>
    </div>
    ${safetyBanner()}
  `;
  el("start-analysis").onclick = () => { STATE.wizard = freshWizard(); navigate("new-analysis"); };
}

// ============================================================ NEW ANALYSIS
async function ensureMeta() {
  if (!STATE.meta) {
    const [specimens, history] = await Promise.all([
      api("/api/specimens"), api("/api/history-fields"),
    ]);
    STATE.meta = { ...specimens, historyFields: history.fields };
    STATE.wizard.ctx.ageBand = specimens.age_bands[5];
  }
}

function stepperHtml(current) {
  const steps = ["Identify Organism", "Clinical Context", "Analyze"];
  return `<div class="stepper">${steps.map((s, i) => {
    const cls = i === current ? "active" : (i < current ? "done" : "");
    return `<div class="step ${cls}">${i + 1}. ${escapeHtml(s)}</div>`;
  }).join("")}</div>`;
}

async function renderNewAnalysis() {
  await ensureMeta();
  const root = el("view-root");
  const w = STATE.wizard;
  root.innerHTML = `<h1>New Clinical Analysis</h1>${stepperHtml(w.step)}<div id="wizard-body"></div>`;
  if (w.step === 0) await renderWizardStep1();
  else if (w.step === 1) await renderWizardStep2();
  else await renderWizardStep3();
}

async function renderWizardStep1() {
  const w = STATE.wizard;
  const body = el("wizard-body");
  body.innerHTML = `
    <h3>Identify Organism</h3>
    <div class="field">
      <label class="field-label">How was the organism identified?</label>
      <div class="checkbox-row"><label><input type="radio" name="mode" value="search" ${w.mode === "search" ? "checked" : ""}> Search bacterial organism</label></div>
      <div class="checkbox-row"><label><input type="radio" name="mode" value="sample" ${w.mode === "sample" ? "checked" : ""}> Load a real de-identified culture (demonstration)</label></div>
    </div>
    <div id="mode-body"></div>
    <div id="organism-confirm"></div>
    <div style="margin-top:1.2rem; display:flex; gap:0.6rem;">
      <button class="btn primary" id="step1-next" disabled>Next</button>
    </div>
  `;
  body.querySelectorAll('input[name=mode]').forEach((r) => {
    r.onchange = () => { w.mode = r.value; renderWizardStep1(); };
  });
  el("step1-next").onclick = () => { w.step = 1; renderNewAnalysis(); };

  const confirmBox = () => {
    const organism = w.mode === "search" ? w.organism : w.sampleOrganism;
    el("organism-confirm").innerHTML = organism
      ? `<div class="banner info compact">Organism: <strong>${escapeHtml(organism)}</strong></div>` : "";
    el("step1-next").disabled = !organism;
  };

  if (w.mode === "search") {
    el("mode-body").innerHTML = `
      <div class="field">
        <label class="field-label">Search bacterial organism&hellip;</label>
        <input type="search" id="organism-search" placeholder="e.g. Escherichia coli, Klebsiella, Staphylococcus…" value="${escapeHtml(w.query)}">
        <div class="field-help">Matches are restricted to the organisms the trained model actually learned from during training.</div>
      </div>
      <div id="organism-results"></div>
    `;
    const resultsBox = el("organism-results");
    const runSearch = async (q) => {
      w.query = q;
      if (!q.trim()) { resultsBox.innerHTML = ""; w.organism = null; confirmBox(); return; }
      resultsBox.innerHTML = loadingHTML("Searching…");
      const res = await api(`/api/organisms?q=${encodeURIComponent(q)}`);
      w.matches = res.matches;
      if (res.total === 0) {
        w.organism = null;
        resultsBox.innerHTML = `<div class="banner error">
          <strong>This organism is not supported by the current trained model.</strong><br>
          Recommended action: select an organism included in the supported, dataset-defined
          vocabulary. Try a shorter or differently spelled search term.</div>`;
      } else {
        const shown = res.matches.slice(0, 25);
        resultsBox.innerHTML = `<div class="field-help">${res.total} matching organism(s)${res.total > 25 ? " — showing first 25" : ""}</div>
          <div class="card" style="max-height:260px; overflow-y:auto;">
          ${shown.map((o) => `<div class="checkbox-row" style="padding:0.25rem 0;">
            <label><input type="radio" name="organism-pick" value="${escapeHtml(o)}" ${w.organism === o ? "checked" : ""}> ${escapeHtml(o)}</label>
          </div>`).join("")}</div>`;
        resultsBox.querySelectorAll('input[name="organism-pick"]').forEach((r) => {
          r.onchange = () => { w.organism = r.value; confirmBox(); };
        });
      }
      confirmBox();
    };
    el("organism-search").addEventListener("input", debounce((e) => runSearch(e.target.value), 250));
    if (w.query) runSearch(w.query);
  } else {
    el("mode-body").innerHTML = loadingHTML("Loading held-out example cases…");
    if (!w.sampleOrders) w.sampleOrders = (await api("/api/sample-orders")).orders;
    const renderOrderPicker = () => {
      el("mode-body").innerHTML = `
        <div class="field">
          <label class="field-label">Held-out culture order</label>
          <select id="sample-order">${w.sampleOrders.map((o) =>
            `<option value="${escapeHtml(o.order_id)}">${escapeHtml(o.organism)} · ${escapeHtml(o.specimen)} · ${escapeHtml(o.age)} · ${o.n_agents_tested} agents tested · ${o.n_nonsusceptible} non-susceptible</option>`
          ).join("")}</select>
          <div class="field-help">This case is a real, de-identified culture from the model's held-out test partition — patients the model never saw during training.</div>
        </div>
        <div id="sample-organism-picker"></div>
      `;
      const sel = el("sample-order");
      const loadOrganisms = async () => {
        w.sampleOrderId = sel.value;
        const res = await api(`/api/sample-orders/${encodeURIComponent(sel.value)}/organisms`);
        w.sampleOrganisms = res.organisms;
        w.sampleOrganism = res.organisms[0];
        if (res.organisms.length > 1) {
          el("sample-organism-picker").innerHTML = `
            <div class="field">
              <label class="field-label">This culture grew ${res.organisms.length} organisms — choose one</label>
              <select id="sample-organism">${res.organisms.map((o) => `<option value="${escapeHtml(o)}">${escapeHtml(o)}</option>`).join("")}</select>
            </div>`;
          el("sample-organism").onchange = (e) => { w.sampleOrganism = e.target.value; confirmBox(); };
        } else {
          el("sample-organism-picker").innerHTML = "";
        }
        confirmBox();
      };
      sel.onchange = loadOrganisms;
      loadOrganisms();
    };
    renderOrderPicker();
  }
  confirmBox();
}

async function renderWizardStep2() {
  const w = STATE.wizard;
  const body = el("wizard-body");
  if (w.mode === "sample") {
    body.innerHTML = `
      <h3>Clinical Context</h3>
      <p class="sub">Organism: <strong>${escapeHtml(w.sampleOrganism)}</strong></p>
      <div class="banner info compact">Clinical context for this case comes directly from the real held-out culture record and does not require manual entry.</div>
      <div style="margin-top:1rem; display:flex; gap:0.6rem;">
        <button class="btn" id="step2-back">Back</button>
        <button class="btn primary" id="step2-next">Next</button>
      </div>`;
    el("step2-back").onclick = () => { w.step = 0; renderNewAnalysis(); };
    el("step2-next").onclick = () => { w.step = 2; renderNewAnalysis(); };
    return;
  }

  const m = STATE.meta;
  const ctx = w.ctx;
  body.innerHTML = `
    <h3>Clinical Context</h3>
    <p class="sub">Organism: <strong>${escapeHtml(w.organism)}</strong></p>

    <div class="field-group">
      <h4>Microbiology</h4>
      <div class="field-row">
        <div class="field">
          <label class="field-label">Specimen type</label>
          <select id="ctx-specimen">${m.specimens.map((s) => `<option ${ctx.specimen === s ? "selected" : ""}>${s}</option>`).join("")}</select>
          <div class="field-help">Used by the model (<code>culture_description</code>).</div>
        </div>
        <div class="field">
          <label class="field-label">Colony morphotype</label>
          <select id="ctx-morphotype">${m.morphotypes.map((s) => `<option ${ctx.morphotype === s ? "selected" : ""}>${s}</option>`).join("")}</select>
          <div class="field-help">Used by the model (<code>organism_morphotype</code>).</div>
        </div>
      </div>
      <div class="field">
        <label class="field-label">Culture year</label>
        <input type="number" id="ctx-year" min="2007" max="2030" value="${ctx.order_year}">
        <div class="field-help">Used by the model (<code>order_year</code>).</div>
      </div>
    </div>

    <div class="field-group">
      <h4>Clinical History</h4>
      <p class="small muted">Only the history fields the deployed model actually uses are collected here.</p>
      <div class="checkbox-row" style="margin-bottom:0.9rem;">
        <label><input type="checkbox" id="ctx-history-unknown" ${ctx.historyUnknown ? "checked" : ""}> Patient history is unknown / unavailable</label>
      </div>
      <div id="history-fields" class="field-row" style="${ctx.historyUnknown ? "display:none;" : ""}">
        ${m.historyFields.map((f) => `
          <div class="field">
            <label class="field-label">${escapeHtml(f.label)}</label>
            <input type="number" class="hist-field" data-key="${f.key}" min="${f.min}" max="${f.max}" value="${ctx.history[f.key] ?? f.default}">
          </div>`).join("")}
      </div>
      <p class="field-help ${ctx.historyUnknown ? "" : "hidden"}" id="history-unknown-note">History fields will be defaulted and reported as assumed.</p>
    </div>

    <div class="field-group">
      <h4>Patient Context</h4>
      <div class="field-row">
        <div class="field">
          <label class="field-label">Recorded binary sex code</label>
          <select id="ctx-gender">
            ${["Not recorded", "0", "1"].map((g) => `<option ${ctx.genderChoice === g ? "selected" : ""}>${g}</option>`).join("")}
          </select>
          <div class="field-help">Used by the model (<code>gender_code</code>). The source data de-identifies sex to codes 0 and 1 with no legend — the mapping to sex is unknown and is never asserted by this application.</div>
        </div>
        <div class="field">
          <label class="field-label reference-only">Care setting (reference only)</label>
          <select id="ctx-setting">${m.settings.map((s) => `<option ${ctx.setting === s ? "selected" : ""}>${s}</option>`).join("")}</select>
          <div class="field-help">Reference only — not used by the resistance prediction model.</div>
        </div>
      </div>
      <div class="field">
        <label class="field-label reference-only">Age band (reference only)</label>
        <select id="ctx-age">${m.age_bands.map((a) => `<option ${ctx.ageBand === a ? "selected" : ""}>${a}</option>`).join("")}</select>
        <div class="field-help">Reference only — no feature in the final model represents patient age.</div>
      </div>

      <div class="expander" id="labs-expander">
        <div class="expander-head"><span>Laboratory results (14-day window) — reference only</span><span class="expander-caret">&#9656;</span></div>
        <div class="expander-body">
          <p class="field-help">Not used by the resistance prediction model. Collected only to give renal context on the Dosage reference in your results.</p>
          <div class="field-row">
            ${[["White cell count (x10^9/L)", "median_wbc"], ["Creatinine (mg/dL)", "median_cr"],
              ["Urea nitrogen (mg/dL)", "median_bun"], ["Haemoglobin (g/dL)", "median_hgb"],
              ["Platelets (x10^9/L)", "median_plt"], ["Sodium (mmol/L)", "median_na"],
              ["Lactate (mmol/L)", "median_lactate"]].map(([label, key]) => `
              <div class="field">
                <label class="field-label">${label}</label>
                <input type="text" class="lab-field" data-key="${key}" value="${ctx.labs[key] ?? ""}">
              </div>`).join("")}
          </div>
        </div>
      </div>

      <div class="field" style="margin-top:1rem;">
        <label class="field-label">Documented allergies (comma separated)</label>
        <input type="text" id="ctx-allergies" placeholder="e.g. penicillin" value="${escapeHtml(ctx.allergiesRaw)}">
        <div class="field-help">Allergies are clinician-entered and used only to raise a flag. They never change a model prediction.</div>
      </div>
    </div>

    <div style="display:flex; gap:0.6rem;">
      <button class="btn" id="step2-back">Back</button>
      <button class="btn primary" id="step2-next">Next</button>
    </div>
  `;

  body.querySelector("#labs-expander .expander-head").onclick = () =>
    body.querySelector("#labs-expander").classList.toggle("open");

  el("ctx-history-unknown").onchange = (e) => {
    ctx.historyUnknown = e.target.checked;
    el("history-fields").style.display = ctx.historyUnknown ? "none" : "";
    el("history-unknown-note").classList.toggle("hidden", !ctx.historyUnknown);
  };

  el("step2-back").onclick = () => { w.step = 0; renderNewAnalysis(); };
  el("step2-next").onclick = () => {
    ctx.specimen = el("ctx-specimen").value;
    ctx.morphotype = el("ctx-morphotype").value;
    ctx.order_year = parseInt(el("ctx-year").value, 10);
    ctx.genderChoice = el("ctx-gender").value;
    ctx.setting = el("ctx-setting").value;
    ctx.ageBand = el("ctx-age").value;
    ctx.allergiesRaw = el("ctx-allergies").value;
    body.querySelectorAll(".hist-field").forEach((inp) => {
      ctx.history[inp.dataset.key] = parseFloat(inp.value || "0");
    });
    body.querySelectorAll(".lab-field").forEach((inp) => {
      ctx.labs[inp.dataset.key] = inp.value.trim() ? parseFloat(inp.value) : null;
    });
    w.step = 2;
    renderNewAnalysis();
  };
}

async function renderWizardStep3() {
  const w = STATE.wizard;
  const body = el("wizard-body");
  const organism = w.mode === "sample" ? w.sampleOrganism : w.organism;
  body.innerHTML = `
    <h3>Analyze</h3>
    <div class="card">
      <p><strong>Clinical information validated</strong></p>
      <p>Organism: <strong>${escapeHtml(organism)}</strong></p>
      ${w.mode === "sample"
        ? `<p>Source: real de-identified held-out culture</p>`
        : `<p>Specimen: <strong>${escapeHtml(w.ctx.specimen)}</strong></p>
           <p>Colony morphotype: <strong>${escapeHtml(w.ctx.morphotype)}</strong></p>
           <p>Culture year: <strong>${w.ctx.order_year}</strong></p>`}
    </div>
    <div style="display:flex; gap:0.6rem;">
      <button class="btn" id="step3-back">Back</button>
      <button class="btn primary" id="step3-analyze">Analyze Resistance Profile</button>
    </div>
    <div id="status-box"></div>
  `;
  el("step3-back").onclick = () => { w.step = 1; renderNewAnalysis(); };
  el("step3-analyze").onclick = runAnalysis;
}

async function runAnalysis() {
  const w = STATE.wizard;
  const box = el("status-box");
  box.innerHTML = `
    <div class="status-box">
      <div class="status-title">Analyzing Antimicrobial Resistance Profile</div>
      <div class="status-line done">&#10003; Validating inputs</div>
      <div class="status-line done">&#10003; Preparing model features</div>
      <div class="status-line active" id="status-predict"><span class="spinner"></span> Predicting resistance probabilities &amp; generating explainability analysis</div>
    </div>`;
  el("step3-analyze").disabled = true;
  try {
    let resp;
    if (w.mode === "sample") {
      resp = await api("/api/analyze-sample", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ order_id: w.sampleOrderId, organism: w.sampleOrganism }),
      });
    } else {
      const ctx = w.ctx;
      const genderCode = ctx.genderChoice === "Not recorded" ? null : parseInt(ctx.genderChoice, 10);
      const allergies = ctx.allergiesRaw.split(",").map((a) => a.trim()).filter(Boolean);
      resp = await api("/api/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          organism: w.organism, specimen: ctx.specimen, morphotype: ctx.morphotype,
          order_year: ctx.order_year, gender_code: genderCode, setting: ctx.setting,
          age_band: ctx.ageBand, history: ctx.history, history_unknown: ctx.historyUnknown,
          labs: ctx.labs, allergies,
        }),
      });
    }
    el("status-predict").innerHTML = "&#10003; Predicting resistance probabilities";
    el("status-predict").classList.replace("active", "done");
    const done = document.createElement("div");
    done.className = "status-line done";
    done.innerHTML = "&#10003; Generating explainability analysis";
    box.querySelector(".status-box").appendChild(done);
    STATE.analysis = resp;
    STATE.detailCache = { dosage: {}, cost: {}, genomic: null, patientHistory: null, groundTruth: null };
    setTimeout(() => navigate("results"), 300);
  } catch (err) {
    box.innerHTML = errorBanner(err);
    el("step3-analyze").disabled = false;
  }
}

// =================================================================== RESULTS
async function renderResults() {
  const root = el("view-root");
  const a = STATE.analysis;
  if (!a) {
    root.innerHTML = `
      <div class="empty-state">
        <h3>No analysis has been run yet.</h3>
        <p>Start a new clinical analysis to identify an organism, enter clinical context, and generate a resistance profile.</p>
        <button class="btn primary" id="empty-start">Start New Analysis</button>
      </div>`;
    el("empty-start").onclick = () => { STATE.wizard = freshWizard(); navigate("new-analysis"); };
    return;
  }

  const options = a.options;
  root.innerHTML = `
    <h1>Antimicrobial Resistance Analysis</h1>
    <div class="metric-row">
      ${metric("Organism", a.organism)}
      ${metric("Analysis Status", "Complete")}
    </div>
    <p class="sub">Specimen: ${escapeHtml(a.specimen || "—")}</p>
    ${safetyBanner(true)}
    ${a.assumed_fields && a.assumed_fields.length ? `
      <div class="banner">
        <strong>Fields not supplied — a training-set default was used.</strong>
        The prediction is correspondingly less specific to this patient.<br>
        ${a.assumed_fields.map((f) => `<code>${escapeHtml(f)}</code>`).join(" ")}
      </div>` : ""}

    <h2>AI Prediction Summary</h2>
    <p>The model has generated predicted resistance probabilities for supported antimicrobial agents.</p>
    <blockquote class="small muted">${escapeHtml(a.framing)}<br><br>${escapeHtml(a.safety_notice)}</blockquote>

    <h2>Antimicrobial Ranking</h2>
    <p class="small muted">Ranked by predicted susceptibility. A lower predicted resistance risk is not an
      instruction to prescribe — every option requires clinician review, and flagged agents (guardrails below)
      are ranked, not removed.</p>
    <div class="table-wrap">
      <table class="data">
        <thead><tr><th>Rank</th><th>Antimicrobial Agent</th><th>Class</th>
          <th>Predicted Resistance Probability</th><th>Risk Category</th><th>Clinical Review</th></tr></thead>
        <tbody>
          ${options.map((o) => `
            <tr data-antibiotic="${escapeHtml(o.antibiotic)}" class="ranking-row">
              <td>${o.rank}</td>
              <td><strong>${escapeHtml(o.antibiotic)}</strong></td>
              <td class="small muted">${escapeHtml(o.antibiotic_subclass || "—")}</td>
              <td><div class="prob-bar-wrap"><div class="prob-bar"><span style="width:${(o.p_nonsusceptible * 100).toFixed(1)}%"></span></div>
                  <span class="prob-value">${fmtPct(o.p_nonsusceptible)}</span></div></td>
              <td>${riskBadge(o.p_nonsusceptible)}</td>
              <td>${o.has_critical_flag ? "&#128721; Guardrail flag" : "No guardrail flags"}</td>
            </tr>`).join("")}
        </tbody>
      </table>
    </div>

    <h2>Options for review — Antibiotic Detail View</h2>
    <p class="small muted">Select an agent below to review the AI's explanation, evidence, and reference
      information for that option. These are options for clinician review, not a prescription.</p>
    <div id="detail-cards"></div>
    <div style="margin:0.8rem 0;"><button class="btn sm" id="show-more">Show more options</button></div>

    <hr>
    <div class="expander" id="exp-patient-history">
      <div class="expander-head"><span>Patient history used by the model</span><span class="expander-caret">&#9656;</span></div>
      <div class="expander-body"><div class="loading-inline">Click to load</div></div>
    </div>
    <div class="expander" id="exp-genomic">
      <div class="expander-head"><span>Genomic — gene-based AMR analysis</span><span class="expander-caret">&#9656;</span></div>
      <div class="expander-body"><div class="loading-inline">Click to load</div></div>
    </div>
    ${a.has_ground_truth ? `
    <div class="expander" id="exp-ground-truth">
      <div class="expander-head"><span>Ground truth for this held-out culture (demonstration)</span><span class="expander-caret">&#9656;</span></div>
      <div class="expander-body"><div class="loading-inline">Click to load</div></div>
    </div>` : ""}

    <hr>
    <h2>Clinician Review — the decision stays with you</h2>
    <pre>AI prediction  &rarr;  Explanation  &rarr;  Potential options  &rarr;  CLINICIAN REVIEW
                                                              &darr;
                                             Accept / Modify / Reject  &rarr;  Notes</pre>
    <p class="small muted">Nothing is actioned by this application. The record below is a review artefact for audit and for the research report.</p>
    <div id="review-form" class="card"></div>
    <div id="review-result"></div>

    <hr>
    <p class="small muted">${escapeHtml(SAFETY_DISCLAIMER)}</p>
  `;

  let shown = 8;
  const renderCards = () => {
    el("detail-cards").innerHTML = options.slice(0, shown).map(cardHtml).join("");
    wireCards();
    el("show-more").style.display = shown >= options.length ? "none" : "";
  };
  el("show-more").onclick = () => { shown = Math.min(shown + 8, options.length); renderCards(); };
  renderCards();

  wireExpander("exp-patient-history", loadPatientHistory);
  wireExpander("exp-genomic", loadGenomic);
  if (a.has_ground_truth) wireExpander("exp-ground-truth", loadGroundTruth);

  renderReviewForm(options);
}

function wireExpander(id, loader) {
  const box = el(id);
  if (!box) return;
  let loaded = false;
  box.querySelector(".expander-head").onclick = async () => {
    box.classList.toggle("open");
    if (box.classList.contains("open") && !loaded) {
      loaded = true;
      await loader(box.querySelector(".expander-body"));
    }
  };
}

function cardHtml(o) {
  const pills = awarePill(o.aware_category) + evidencePill(o.evidence_strength) +
    (o.route_oral_available ? `<span class="pill blue">oral option available</span>` : "") +
    (o.urinary_only ? `<span class="pill neutral">urinary agent only</span>` : "");
  return `
    <div class="card ${o.has_critical_flag ? "flagged" : ""}">
      <h4>${o.rank}. ${escapeHtml(o.antibiotic)}
        <span class="small muted" style="font-weight:400;"> — ${escapeHtml(o.antibiotic_subclass || "")}</span></h4>
      <div style="margin-bottom:0.5rem;">${pills}</div>
      <div class="small">${riskBadge(o.p_nonsusceptible)} &middot; predicted susceptibility:
        ${fmtPct(o.p_susceptible)} &middot; ${fmtNum(o.training_examples)} training examples for this organism-agent pair</div>
      <div class="expander" data-antibiotic="${escapeHtml(o.antibiotic)}" style="margin-top:0.7rem;">
        <div class="expander-head"><span>Detail — ${escapeHtml(o.antibiotic)}</span><span class="expander-caret">&#9656;</span></div>
        <div class="expander-body"></div>
      </div>
    </div>`;
}

function wireCards() {
  const a = STATE.analysis;
  document.querySelectorAll("#detail-cards .expander").forEach((box) => {
    const antibiotic = box.dataset.antibiotic;
    const o = a.options.find((x) => x.antibiotic === antibiotic);
    let built = false;
    box.querySelector(".expander-head").onclick = () => {
      box.classList.toggle("open");
      if (box.classList.contains("open") && !built) {
        built = true;
        buildDetailPanel(box.querySelector(".expander-body"), o);
      }
    };
  });
}

function buildDetailPanel(container, o) {
  container.innerHTML = `
    <div class="tabs">
      <button class="tab-btn active" data-tab="clinical">Clinical View</button>
      <button class="tab-btn" data-tab="technical">Technical Details</button>
      <button class="tab-btn" data-tab="dosage">Dosage reference</button>
      <button class="tab-btn" data-tab="cost">Cost</button>
    </div>
    <div class="tab-panel active" data-panel="clinical">${clinicalPanelHtml(o)}</div>
    <div class="tab-panel" data-panel="technical">${technicalPanelHtml(o)}</div>
    <div class="tab-panel" data-panel="dosage"><div class="loading-inline">Click to load</div></div>
    <div class="tab-panel" data-panel="cost"><div class="loading-inline">Click to load</div></div>
  `;
  const tabs = container.querySelectorAll(".tab-btn");
  const panels = container.querySelectorAll(".tab-panel");
  const loaded = { dosage: false, cost: false };
  tabs.forEach((btn) => {
    btn.onclick = async () => {
      tabs.forEach((b) => b.classList.remove("active"));
      panels.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const name = btn.dataset.tab;
      container.querySelector(`.tab-panel[data-panel="${name}"]`).classList.add("active");
      if (name === "dosage" && !loaded.dosage) { loaded.dosage = true; await loadDosage(container, o); }
      if (name === "cost" && !loaded.cost) { loaded.cost = true; await loadCost(container, o); }
    };
  });
}

function clinicalPanelHtml(o) {
  const up = (o.top_factors || []).filter((f) => f.shap > 0);
  const down = (o.top_factors || []).filter((f) => f.shap < 0);
  const humanise = (f) => f.replace(/prior_med_class__/, "prior exposure to ")
    .replace(/prior_res_/, "prior resistance ").replace(/prior_med_/, "prior exposure ")
    .replace(/__/g, " ").replace(/_/g, " ");
  let html = `<p class="small">AI Prediction Rationale — Explanation</p>`;
  html += `<p><strong>Model reasoning.</strong> ${escapeHtml(o.explanation)}</p>`;
  if (o.top_factors && o.top_factors.length) {
    html += `<p><strong>Factors increasing predicted resistance</strong></p>`;
    html += up.length ? `<ul>${up.map((f) => `<li>&#9650; ${escapeHtml(humanise(f.feature))}</li>`).join("")}</ul>`
      : `<p class="small muted">None among the top contributing factors.</p>`;
    html += `<p><strong>Factors decreasing predicted resistance</strong></p>`;
    html += down.length ? `<ul>${down.map((f) => `<li>&#9660; ${escapeHtml(humanise(f.feature))}</li>`).join("")}</ul>`
      : `<p class="small muted">None among the top contributing factors.</p>`;
  } else {
    html += `<div class="banner info compact">Per-factor explanation unavailable for this prediction. SHAP could not be computed for this model/row combination. The probability above is unaffected.</div>`;
  }
  if (o.flags && o.flags.length) {
    html += `<p><strong>Guardrails:</strong></p>` + o.flags.map(flagBox).join("");
  } else {
    html += `<p class="small muted">No guardrail flags raised for this agent.</p>`;
  }
  return html;
}

function technicalPanelHtml(o) {
  let html = `<p class="small muted">Contributions are SHAP values computed on the deployed model for this specific isolate — not generic feature importances.</p>`;
  if (o.top_factors && o.top_factors.length) {
    html += `<div class="table-wrap"><table class="data"><thead><tr>
      <th>Model feature</th><th>Encoded value</th><th>Effect on predicted resistance</th><th>Direction</th>
      </tr></thead><tbody>${o.top_factors.map((f) => `
        <tr><td>${escapeHtml(f.feature)}</td><td>${f.value === null ? "—" : f.value}</td>
        <td>${f.shap.toFixed(4)}</td><td class="small">${escapeHtml(f.direction)}</td></tr>`).join("")}
      </tbody></table></div>
      <p class="small muted">Positive values push the prediction towards resistance; negative values push it
      towards susceptibility. Values are on the model's log-odds scale.</p>`;
  } else {
    html += `<p class="small muted">No SHAP values available for this row.</p>`;
  }
  html += `<div class="metric-row" style="margin-top:0.8rem;">
    ${metric("Confidence band", o.confidence_band)}
    ${metric("Training examples", fmtNum(o.training_examples))}</div>`;
  return html;
}

async function loadDosage(container, o) {
  const panel = container.querySelector('.tab-panel[data-panel="dosage"]');
  panel.innerHTML = loadingHTML("Loading dosage reference…");
  try {
    const cacheKey = o.antibiotic;
    if (!STATE.detailCache.dosage[cacheKey]) {
      STATE.detailCache.dosage[cacheKey] = await api(
        `/api/analysis/${STATE.analysis.analysis_id}/dosage?antibiotic=${encodeURIComponent(o.antibiotic)}`);
    }
    const d = STATE.detailCache.dosage[cacheKey];
    if (!d.available) {
      panel.innerHTML = `<p class="small">Dosage reference — clinician verification required</p>
        <div class="banner info compact"><strong>${escapeHtml(d.message)}</strong>${d.reason ? `<br>${escapeHtml(d.reason)}` : ""}</div>`;
      return;
    }
    panel.innerHTML = `
      <p class="small">Dosage reference — clinician verification required</p>
      <div class="banner compact">${escapeHtml(d.clinician_notice)}</div>
      <p><strong>Dosage and administration (FDA label, verbatim)</strong></p>
      <pre style="white-space:pre-wrap;">${escapeHtml(d.dosage_and_administration)}</pre>
      <p class="small muted">Source: ${escapeHtml(d.source || "—")} ${d.retrieved_at_utc ? `&middot; retrieved ${escapeHtml(d.retrieved_at_utc)} UTC` : ""}</p>
    `;
  } catch (err) { panel.innerHTML = errorBanner(err); }
}

async function loadCost(container, o) {
  const panel = container.querySelector('.tab-panel[data-panel="cost"]');
  panel.innerHTML = loadingHTML("Loading cost reference…");
  try {
    const cacheKey = o.antibiotic;
    if (!STATE.detailCache.cost[cacheKey]) {
      STATE.detailCache.cost[cacheKey] = await api(
        `/api/analysis/${STATE.analysis.analysis_id}/cost?antibiotic=${encodeURIComponent(o.antibiotic)}`);
    }
    const c = STATE.detailCache.cost[cacheKey];
    if (!c.available) {
      panel.innerHTML = `<p class="small">Medicine cost reference</p>
        <div class="banner info compact">Price unavailable, verify locally.${c.reason ? `<br>${escapeHtml(c.reason)}` : ""}</div>`;
      return;
    }
    panel.innerHTML = `
      <p class="small">Medicine cost reference</p>
      <div class="table-wrap"><table class="data"><thead><tr><th>Product</th><th>Price (USD)</th><th>Per</th><th>Generic/Brand</th></tr></thead>
        <tbody>${c.products.slice(0, 3).map((p) => `<tr><td>${escapeHtml(p.description)}</td>
          <td>${p.price_per_unit_usd}</td><td>${escapeHtml(p.pricing_unit)}</td><td>${escapeHtml(p.generic_or_brand)}</td></tr>`).join("")}
      </tbody></table></div>
      <div class="banner compact">${escapeHtml(c.cost_notice)}</div>
      <p class="small muted">Source: ${escapeHtml(c.source || "—")}</p>
    `;
  } catch (err) { panel.innerHTML = errorBanner(err); }
}

async function loadPatientHistory(container) {
  container.innerHTML = loadingHTML();
  try {
    const d = await api(`/api/analysis/${STATE.analysis.analysis_id}/patient-history`);
    container.innerHTML = `
      <p class="small">Patient history — restricted to events strictly BEFORE this culture.</p>
      <div class="field-row">
        <div>
          <h4>Prior cultures, resistance and antibiotic exposure</h4>
          ${simpleTable(d.history, ["History item", "Value"])}
        </div>
        <div>
          <h4>Laboratory results (14 days before the culture)</h4>
          ${simpleTable(d.labs, ["Analyte", "Median over 14 days before culture", "Unit"])}
        </div>
      </div>`;
  } catch (err) { container.innerHTML = errorBanner(err); }
}

function simpleTable(rows, cols) {
  if (!rows || !rows.length) return `<p class="small muted">No data.</p>`;
  const keys = Object.keys(rows[0]).filter((k) => k !== "Field");
  return `<div class="table-wrap"><table class="data"><thead><tr>${keys.map((k) => `<th>${escapeHtml(k)}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${keys.map((k) => `<td>${escapeHtml(r[k])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

async function loadGenomic(container) {
  container.innerHTML = loadingHTML();
  try {
    const d = await api(`/api/analysis/${STATE.analysis.analysis_id}/genomic`);
    if (!d.available) {
      container.innerHTML = `
        <p class="small">Genomic AMR marker analysis</p>
        <div class="banner error"><strong>Genomic module inactive — no genomic data exists in this project's datasets.</strong><br>${escapeHtml(d.reason)}</div>
        <p class="small muted">A reference map of ${d.reference_map_size} AMR markers is implemented and ready. ${escapeHtml(d.reference_note || "")}</p>
        <hr>
        <p><strong>Demonstration of the annotation output</strong></p>
        <p class="small muted">Illustrative only — these markers are NOT from this patient or this dataset.</p>
        <div id="genomic-demo-picker">
          ${d.demo_markers.map((m) => `<label class="checkbox-row" style="display:inline-flex; margin-right:1rem;">
            <input type="checkbox" class="genomic-demo-cb" value="${escapeHtml(m)}" ${["blaKPC-2", "mecA"].includes(m) ? "checked" : ""}> ${escapeHtml(m)}</label>`).join("")}
        </div>
        <div id="genomic-demo-result"></div>`;
      const runDemo = async () => {
        const chosen = Array.from(container.querySelectorAll(".genomic-demo-cb:checked")).map((c) => c.value);
        const out = el ? container.querySelector("#genomic-demo-result") : null;
        if (!chosen.length) { out.innerHTML = ""; return; }
        out.innerHTML = loadingHTML();
        const res = await api("/api/genomic/annotate", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ markers: chosen }),
        });
        out.innerHTML = res.markers.map((a) => `
          <div class="expander open" style="margin-top:0.6rem;"><div class="expander-head"><span>${escapeHtml(a.marker)} — ${escapeHtml(a.gene_family)}</span></div>
          <div class="expander-body" style="display:block;">
            <p><strong>Mechanism.</strong> ${escapeHtml(a.mechanism)}</p>
            <p class="small">Antibiotic classes affected: ${escapeHtml((a.affected_classes || []).join(", ") || "—")}</p>
            <p class="small">Typical organisms: ${escapeHtml((a.typical_organisms || []).join(", ") || "—")}</p>
            <div class="banner info compact">${escapeHtml(a.implication)}</div>
            <p class="small muted">Confidence: ${escapeHtml(a.confidence)} &middot; Reference: ${escapeHtml(a.reference)}</p>
          </div></div>`).join("");
      };
      container.querySelectorAll(".genomic-demo-cb").forEach((cb) => cb.onchange = runDemo);
      runDemo();
    } else {
      if (!d.markers.length) {
        container.innerHTML = `<div class="banner info compact">A genotype table is present but contains no markers for this culture order.</div>`;
        return;
      }
      container.innerHTML = d.markers.map((a) => `
        <div class="expander open" style="margin-bottom:0.6rem;"><div class="expander-head"><span>${escapeHtml(a.marker)} — ${escapeHtml(a.gene_family)}</span></div>
        <div class="expander-body" style="display:block;">
          <p><strong>Mechanism.</strong> ${escapeHtml(a.mechanism)}</p>
          <p class="small">Antibiotic classes affected: ${escapeHtml((a.affected_classes || []).join(", "))}</p>
          <div class="banner info compact">${escapeHtml(a.implication)}</div>
          <p class="small muted">Reference: ${escapeHtml(a.reference)}</p>
        </div></div>`).join("");
    }
  } catch (err) { container.innerHTML = errorBanner(err); }
}

async function loadGroundTruth(container) {
  container.innerHTML = loadingHTML();
  try {
    const d = await api(`/api/analysis/${STATE.analysis.analysis_id}/ground-truth`);
    const agree = d.agreement;
    container.innerHTML = `
      <p class="small muted">The laboratory's actual AST result, shown so the prediction can be checked. It is never used as an input.</p>
      <div class="table-wrap"><table class="data"><thead><tr><th>Antibiotic</th><th>Laboratory result</th><th>Model P(susceptible)</th><th>Model call</th><th>Agreement</th></tr></thead>
      <tbody>${d.rows.map((r) => `<tr><td>${escapeHtml(r.antibiotic)}</td><td>${escapeHtml(r.laboratory_result)}</td>
        <td>${fmtPct(r.p_susceptible, 1)}</td><td>${escapeHtml(r.predicted_label || "—")}</td>
        <td>${r.correct === "agree" ? "&#10003;" : (r.correct === "disagree" ? "&#10007;" : "—")}</td></tr>`).join("")}
      </tbody></table></div>
      ${agree.n_total ? `<div class="banner info compact">Agreement on this culture: <strong>${agree.n_agree}/${agree.n_total}</strong> agents (${fmtPct(agree.rate)}). A single culture is far too small to judge a model — see Model Insights for the held-out test metrics.</div>` : ""}
    `;
  } catch (err) { container.innerHTML = errorBanner(err); }
}

function renderReviewForm(options) {
  const topDefault = options.find((o) => !o.has_critical_flag) || options[0];
  el("review-form").innerHTML = `
    <div class="field-row">
      <div class="field"><label class="field-label">Reviewing clinician (name or identifier)</label>
        <input type="text" id="rev-reviewer"></div>
      <div class="field"><label class="field-label">Decision on the model's ranking</label>
        <select id="rev-decision"><option>Accept</option><option>Modify</option><option>Reject</option></select></div>
    </div>
    <div class="field">
      <label class="field-label">Agent(s) the clinician considers appropriate</label>
      <select id="rev-agents" multiple size="6" style="height:auto;">
        ${options.map((o) => `<option value="${escapeHtml(o.antibiotic)}" ${o === topDefault ? "selected" : ""}>${escapeHtml(o.antibiotic)}</option>`).join("")}
      </select>
    </div>
    <div class="field"><label class="field-label">Clinical reasoning / notes</label>
      <textarea id="rev-notes" placeholder="e.g. Model ranks nitrofurantoin first, but this is a bacteraemic urinary source, so a systemic agent is required."></textarea></div>
    <div class="checkbox-row" style="margin-bottom:0.9rem;">
      <label><input type="checkbox" id="rev-confirm"> I confirm I am a qualified healthcare professional and that this decision is my own clinical judgement, informed but not determined by the model output.</label>
    </div>
    <button class="btn primary" id="rev-submit">Record review</button>
  `;
  el("rev-submit").onclick = async () => {
    const reviewer = el("rev-reviewer").value;
    const decision = el("rev-decision").value;
    const selected = Array.from(el("rev-agents").selectedOptions).map((o) => o.value);
    const notes = el("rev-notes").value;
    const confirm = el("rev-confirm").checked;
    const out = el("review-result");
    try {
      const res = await api(`/api/analysis/${STATE.analysis.analysis_id}/clinician-review`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reviewer, decision, selected_agents: selected, notes, confirm }),
      });
      out.innerHTML = `<div class="banner compact">Review recorded and appended to <code>${escapeHtml(res.path)}</code>.</div>`;
    } catch (err) { out.innerHTML = errorBanner(err); }
  };
}

// ========================================================= MODEL INSIGHTS
async function renderModelInsights() {
  if (!STATE.insights) STATE.insights = await api("/api/model-insights");
  const d = STATE.insights;
  const meta = d.meta || {};
  const root = el("view-root");
  root.innerHTML = `
    <h1>Model Insights</h1>
    ${safetyBanner()}

    <h2>Model development</h2>
    <p class="small muted">Multiple model families were evaluated on the same feature set and validation partition before selecting a final algorithm.</p>
    <div class="table-wrap">
      <table class="data"><thead><tr><th>Algorithm</th><th>Accuracy</th><th>Precision</th><th>Sensitivity</th>
        <th>Specificity</th><th>F1</th><th>ROC-AUC</th><th>PR-AUC</th><th>Brier</th><th>Train (s)</th></tr></thead>
        <tbody>${(d.model_family_comparison || []).map((r) => `<tr ${r.algorithm === meta.algorithm ? 'style="background:var(--accent-bg);font-weight:600;"' : ""}>
          <td>${escapeHtml(r.algorithm)}</td><td>${r.accuracy}</td><td>${r.precision}</td>
          <td>${r.recall_sensitivity}</td><td>${r.specificity}</td><td>${r.f1}</td>
          <td>${r.roc_auc}</td><td>${r.pr_auc}</td><td>${r.brier}</td><td>${r.train_seconds}</td></tr>`).join("")}
        </tbody></table>
    </div>
    <p class="small muted">Final selected model: <strong>${escapeHtml(meta.algorithm || "—")}</strong> — selected primarily
      on PR-AUC (the metric best suited to this class-imbalanced task), with training time and calibration quality
      as secondary considerations. Source: <code>05_MODEL_TRAINING/experiments/experiment_log.csv</code>.</p>
    <div class="metric-row">
      ${metric("Algorithm", meta.algorithm)}
      ${metric("Features", meta.n_features)}
      ${metric("Training rows", fmtNum(meta.n_train_rows))}
      ${metric("Training patients", fmtNum(meta.n_train_patients))}
    </div>

    <h2>1. Held-out test performance</h2>
    <p class="small muted">Primary held-out metric — a random, patient-grouped split. No patient appears in both train and test.</p>
    ${metricsGrid(d.held_out_test)}

    <h2>2. Temporal holdout performance</h2>
    <p class="small muted">Trained on the past, tested on the future — the deployment-realistic figure. Patients spanning the cutoff are removed from the test side.</p>
    ${metricsGrid(d.temporal_test)}

    <h2>3. Three-class experiment</h2>
    <p class="small muted">A secondary experiment predicting Susceptible / Intermediate / Resistant directly, rather than the primary binary target.</p>
    ${d.three_class ? `
      <div class="metric-row">
        ${metric("Accuracy", d.three_class.accuracy)}
        ${metric("Macro F1", d.three_class.macro_f1)}
        ${metric("Weighted F1", d.three_class.weighted_f1)}
      </div>
      <div class="table-wrap"><table class="data"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
      <tbody>${["Susceptible", "Intermediate", "Resistant"].map((c) => {
        const r = d.three_class.classification_report[c];
        return `<tr><td>${c}</td><td>${r.precision.toFixed(4)}</td><td>${r.recall.toFixed(4)}</td><td>${r["f1-score"].toFixed(4)}</td><td>${r.support}</td></tr>`;
      }).join("")}</tbody></table></div>
      <div class="banner info compact">${escapeHtml(d.three_class.conclusion || "")}</div>
    ` : `<p class="small muted">Not available.</p>`}

    <h2>4. User Acceptance Testing</h2>
    <div class="banner">This section reports clinician-facing acceptance testing, <strong>not</strong> a model accuracy metric.</div>
    ${d.uat ? `
      <div class="metric-row">
        ${metric("Test result", d.uat.result)}
        ${metric("Tests passed", `${d.uat.tests_passed}/${d.uat.tests_total}`)}
        ${metric("Cases scanned", d.uat.cases_scanned)}
        ${metric("Demo-menu agreement", fmtPct(d.uat.demo_menu_agreement_rate, 2))}
      </div>
      <p class="small muted">${escapeHtml(d.uat.demo_menu_caveat || "")}</p>
    ` : `<p class="small muted">Not available.</p>`}

    <h2>Intended use</h2>
    <p><strong>Intended.</strong> A research and educational decision-support prototype that estimates the
    probability a laboratory would report an identified isolate as non-susceptible to a given antibiotic, and
    presents that estimate — with its explanation, its evidence base and its guardrails — to a qualified
    clinician during the window between organism identification and the AST result.</p>
    <p><strong>Not intended.</strong></p>
    <ul>
      <li>Autonomous prescribing, or any use where the output is acted on without a qualified clinician reviewing it.</li>
      <li>A substitute for antimicrobial susceptibility testing.</li>
      <li>Use outside the population it was fitted on (see Limitations).</li>
      <li>Prediction of clinical cure, mortality, or any outcome other than the laboratory susceptibility category.</li>
      <li>Dose selection — the application relays FDA label text verbatim and never computes a dose.</li>
    </ul>

    <h2>Limitations</h2>
    <ul>
      <li>Single-centre, single-country data — all cultures come from one US academic health system.</li>
      <li>Three specimen types only — urine, blood and respiratory.</li>
      <li>Adults only — the youngest age band is 18-24 years.</li>
      <li>Panel-selection bias — an isolate only has a label for the agents the laboratory chose to test.</li>
      <li>Labels are laboratory categories, not outcomes.</li>
      <li>No genomic data — this is an out-of-scope, inactive module.</li>
      <li>Partial history coverage — absence of a record is not evidence of absence of exposure.</li>
      <li>Deidentified sex — no sex-specific claim can be made or checked.</li>
      <li>Temporal drift — performance on a temporal holdout is lower than on a random patient split.</li>
    </ul>

    <h2>Fairness and subgroup performance</h2>
    ${Object.keys(d.subgroups || {}).length
      ? Object.entries(d.subgroups).map(([k, rows]) => `
        <div class="expander" data-sub="${k}"><div class="expander-head"><span>By ${escapeHtml(k)}</span><span class="expander-caret">&#9656;</span></div>
        <div class="expander-body">${simpleTable(rows, [])}</div></div>`).join("")
      : `<p class="small muted">Not available.</p>`}
  `;
  root.querySelectorAll(".expander").forEach((box) => {
    box.querySelector(".expander-head").onclick = () => box.classList.toggle("open");
  });
}

function metricsGrid(m) {
  if (!m) return `<p class="small muted">Not available.</p>`;
  return `<div class="metric-row">
    ${metric("PR-AUC", m.pr_auc)} ${metric("ROC-AUC", m.roc_auc)}
    ${metric("Sensitivity", m.recall_sensitivity)} ${metric("Specificity", m.specificity)}
  </div>
  <div class="metric-row">
    ${metric("Precision (PPV)", m.precision_ppv ?? m.precision)} ${metric("F1", m.f1)}
    ${metric("Accuracy", m.accuracy)} ${metric("Brier score", m.brier)}
  </div>`;
}

// =========================================================== HOW IT WORKS
async function renderHowItWorks() {
  if (!STATE.methodology) STATE.methodology = await api("/api/methodology");
  const d = STATE.methodology;
  const root = el("view-root");
  root.innerHTML = `
    <h1>How It Works</h1>
    ${safetyBanner()}
    <pre>Clinical / Laboratory Data
        &darr;
Feature Processing
        &darr;
LightGBM Prediction
        &darr;
Resistance Probability
        &darr;
Antimicrobial Ranking
        &darr;
SHAP Explanation
        &darr;
Clinical Decision Support</pre>
    <p>This system does not replace:</p>
    <ul><li>Laboratory antimicrobial susceptibility testing</li><li>Clinical expertise</li>
      <li>Medical guidelines</li><li>Clinician judgement</li></ul>
    <p>It provides AI-assisted predictive insight based on learned historical data patterns, for review by a qualified clinician.</p>

    <hr>
    <h2>Dataset and methodology</h2>
    <div class="tabs" id="meth-tabs">
      <button class="tab-btn active" data-t="source">Source data</button>
      <button class="tab-btn" data-t="cleaning">Cleaning decisions</button>
      <button class="tab-btn" data-t="leakage">Leakage controls</button>
      <button class="tab-btn" data-t="pipeline">Pipeline</button>
      <button class="tab-btn" data-t="docs">Documents</button>
    </div>
    <div class="tab-panel active" data-p="source">${methSourceHtml(d)}</div>
    <div class="tab-panel" data-p="cleaning">${methCleaningHtml(d)}</div>
    <div class="tab-panel" data-p="leakage">${methLeakageHtml(d)}</div>
    <div class="tab-panel" data-p="pipeline">${methPipelineHtml()}</div>
    <div class="tab-panel" data-p="docs">${methDocsHtml(d)}</div>
  `;
  const tabs = root.querySelectorAll("#meth-tabs .tab-btn");
  const panels = root.querySelectorAll(".tab-panel[data-p]");
  tabs.forEach((btn) => btn.onclick = () => {
    tabs.forEach((b) => b.classList.remove("active"));
    panels.forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    root.querySelector(`.tab-panel[data-p="${btn.dataset.t}"]`).classList.add("active");
  });
  root.querySelectorAll(".expander").forEach((box) => {
    box.querySelector(".expander-head").onclick = () => box.classList.toggle("open");
  });
}

function methSourceHtml(d) {
  const audit = d.audit, build = d.build;
  let html = `<h3>The five source files</h3>`;
  if (audit) {
    html += `<div class="table-wrap"><table class="data"><thead><tr><th>Dataset</th><th>File</th><th>Size (MB)</th><th>Rows</th><th>Columns</th></tr></thead>
      <tbody>${audit.datasets.map((ds) => `<tr><td>${escapeHtml(ds.key)}</td><td>${escapeHtml(ds.file_name)}</td>
        <td>${ds.file_size_mb.toFixed(1)}</td><td>${fmtNum(ds.n_rows)}</td><td>${ds.n_columns}</td></tr>`).join("")}
      </tbody></table></div>
      <p class="small muted">Raw files are referenced in place and never modified.</p>`;
  }
  if (build) {
    const s1 = build.step1_cohort, f = build.step7_final;
    html += `<h3>From raw rows to the modelling table</h3>
      <div class="table-wrap"><table class="data"><tbody>
        <tr><td>Raw cohort rows</td><td>${fmtNum(s1.n_rows_raw)}</td></tr>
        <tr><td>&minus; negative cultures (no susceptibility result)</td><td>-${fmtNum(s1.n_dropped_no_label_negative_culture)}</td></tr>
        <tr><td>&minus; Inconclusive / Synergism</td><td>-${fmtNum(s1.n_dropped_non_phenotype_result)}</td></tr>
        <tr><td>&minus; duplicate resolution</td><td>-${fmtNum(s1.n_dropped_duplicate_resolution)}</td></tr>
        <tr><td><strong>= modelling rows</strong></td><td><strong>${fmtNum(f.n_rows)}</strong></td></tr>
      </tbody></table></div>
      <div class="metric-row">
        ${metric("Modelling rows", fmtNum(f.n_rows))} ${metric("Culture orders", fmtNum(f.n_orders))}
        ${metric("Patients", fmtNum(f.n_patients))} ${metric("Non-susceptible rate", fmtPct(f.target_positive_rate, 1))}
      </div>`;
  }
  return html;
}

function methCleaningHtml(d) {
  if (!d.build) return `<p class="small muted">Not available.</p>`;
  const s1 = d.build.step1_cohort, s3 = d.build.step3_labs, s4 = d.build.step4_prior_med;
  return `
    <h3>Organism names</h3>
    <p>The raw <code>organism</code> column holds <strong>${fmtNum(s1.organism_raw_distinct)}</strong> distinct strings.
    After normalisation there are <strong>${fmtNum(s1.organism_clean_distinct)}</strong> clean species-level names.</p>
    <h3>Duplicate resolution</h3>
    <p><strong>${fmtNum(s1.n_rows_in_duplicate_key_groups)}</strong> rows sat in
    <strong>${fmtNum(s1.n_duplicate_key_groups)}</strong> duplicated groups. Resolution rule:
    <strong>${escapeHtml(s1.dedup_rule)}</strong>. ${fmtNum(s1.n_dropped_duplicate_resolution)} rows removed.</p>
    <h3>Laboratory results</h3>
    <p>The labs export carries <strong>${s3.rows_per_order.mean}</strong> rows per culture order on average.
    ${escapeHtml(s3.collapse_rule)}</p>
    <h3>Prior medications</h3>
    <p>${escapeHtml(s4.dedup_note)}</p>
    <p><strong>${fmtNum(s4.n_rows_same_day_t_eq_0)}</strong> rows sat at exactly 0 days from the culture.
    ${escapeHtml(s4.same_day_exclusion_reason)}</p>`;
}

function methLeakageHtml(d) {
  if (!d.build || !d.leakage) return `<p class="small muted">Not available.</p>`;
  const s1 = d.build.step1_cohort, rt = d.leakage.resistance_time_to_culture;
  return `
    <div class="banner error"><strong>Resistance phenotype embedded in the organism name — ${fmtNum(s1.organism_leaked_phenotype_rows)} rows.</strong></div>
    <p>Values such as <code>STAPH AUREUS {MRSA}</code> restate the label in the organism field. The normaliser
    strips them into a separate field which is then dropped.</p>
    <div class="banner compact">Prior resistance history — investigated and cleared.</div>
    <p>Minimum time before the culture is <strong>${rt.min} day</strong>, with <strong>${rt.n_le_zero}</strong>
    events at or after the culture — cleared as a leakage risk.</p>
    <div class="banner info compact">Split-level leakage: every split is grouped by patient. No patient appears
    in more than one partition.</div>`;
}

function methPipelineHtml() {
  return `<pre>Raw CSVs (read-only, referenced in place)
        &darr;  scripts/01_dataset_audit.py
Dataset audit
        &darr;  scripts/04_build_analytic_dataset.py
Cleaning + feature engineering
        &darr;  scripts/06_feature_selection.py
Feature selection  &rarr;  config/final_features.json
        &darr;  scripts/07_train_models.py
6 model families, tuning, calibration  &rarr;  final_model.joblib
        &darr;  scripts/08_evaluate_models.py
Held-out test evaluation
        &darr;  scripts/09_explainability.py
SHAP global + local
        &darr;
Recommendation engine  &rarr;  Web application  &rarr;  CLINICIAN</pre>`;
}

function methDocsHtml(d) {
  if (!d.docs || !d.docs.length) return `<p class="small muted">No documents generated yet.</p>`;
  return d.docs.map((doc) => `
    <div class="expander"><div class="expander-head"><span>${escapeHtml(doc.name)}</span><span class="expander-caret">&#9656;</span></div>
    <div class="expander-body"><pre style="white-space:pre-wrap;">${escapeHtml(doc.content)}</pre></div></div>`).join("");
}

// -------------------------------------------------------------------- init
refreshModelStatus();
navigate("dashboard");
