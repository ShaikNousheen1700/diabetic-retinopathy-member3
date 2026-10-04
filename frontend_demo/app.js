/* ==========================================================================
   Website logic.
   API calls (unchanged backend contract):
     GET  /health        -> {status, model_loaded, uptime_seconds}
     GET  /model-info    -> model details shown in "Model information"
     POST /predict       -> multipart/form-data, field "file"
                            success: {success:true, prediction:{class_id, class_name, confidence, probabilities}, ...}
                            error:   {success:false, error:{code, message}}
   Research numbers come from /project-results/densenet121_metrics.json (written by src/evaluate.py).
   Nothing shown on the page is invented: every number is read from the API or the results file.
   ========================================================================== */

const API = "";   // same server that serves this page; e.g. "http://127.0.0.1:8000" if hosted elsewhere
document.documentElement.classList.add("js");

// One shared description of the five grades (used by the levels section and the result card).
// `api` must match the class names returned by the backend.
const GRADES = [
  { id: 0, api: "No DR", name: "No DR", short: "No DR", sub: "No diabetic retinopathy",
    text: "No visible signs of diabetic retinopathy in the retina." },
  { id: 1, api: "Mild", name: "Mild", short: "Mild", sub: "Mild non-proliferative DR",
    text: "Earliest stage. Tiny bulges in small blood vessel walls (microaneurysms) appear." },
  { id: 2, api: "Moderate", name: "Moderate", short: "Moderate", sub: "Moderate non-proliferative DR",
    text: "More vessel damage: small bleeds, yellow fatty deposits and leaking vessels can be seen." },
  { id: 3, api: "Severe", name: "Severe", short: "Severe", sub: "Severe non-proliferative DR",
    text: "Many vessels are blocked or bleeding across the retina, so areas lose their blood supply." },
  { id: 4, api: "Proliferative DR", name: "Proliferative DR", short: "Prolif.", sub: "Proliferative DR",
    text: "Most advanced stage. Fragile new blood vessels grow and can bleed, which can seriously affect vision." },
];

const PROGRESSION = [
  { title: "Healthy retina", text: "Clear blood vessels and optic disc." },
  { title: "Early changes", text: "A few tiny red dots (microaneurysms)." },
  { title: "Increasing damage", text: "Small bleeds and yellow deposits appear." },
  { title: "Severe disease", text: "Widespread bleeding; areas lose blood supply." },
  { title: "Proliferative stage", text: "Fragile new vessels grow and may bleed." },
];

// Verified class counts of the 3,662 APTOS training images (train.csv, checked in notebook 01, section 5)
const CLASS_COUNTS = [1805, 370, 999, 193, 295];

const $ = (sel) => document.querySelector(sel);
const pct = (x, digits = 1) => `${(x * 100).toFixed(digits)}%`;
const gradeVar = (id) => `var(--g${id})`;
const gradeInk = (id) => `var(--gi${id})`;

/* ---------------------------------------------------------------- navigation */
function setupNav() {
  const nav = $(".nav"), toggle = $(".nav-toggle"), links = $("#nav-links");
  toggle.addEventListener("click", () => {
    const open = links.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });
  links.addEventListener("click", (e) => {
    if (e.target.tagName === "A") { links.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); }
  });
  window.addEventListener("scroll", () => nav.classList.toggle("scrolled", window.scrollY > 8), { passive: true });

  // highlight the nav link of the section currently on screen
  const anchors = [...links.querySelectorAll("a")];
  const sections = anchors.map((a) => document.querySelector(a.getAttribute("href")));
  const spy = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (en.isIntersecting) {
        anchors.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + en.target.id));
      }
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  sections.forEach((s) => s && spy.observe(s));
}

function setupReveal() {
  const items = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) { items.forEach((el) => el.classList.add("in")); return; }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } });
  }, { threshold: 0.12 });
  items.forEach((el) => io.observe(el));
}

/* ---------------------------------------------------------------- educational content */
function renderEducation() {
  $("#hero-retina").innerHTML = retinaSVG(0, { label: "Illustration of a healthy retina" });
  $("#empty-retina").innerHTML = retinaSVG(0);

  $("#levels-list").innerHTML = GRADES.map((g) => `
    <li class="level reveal" style="--g:${gradeVar(g.id)};--gi:${gradeInk(g.id)}">
      <span class="level-num">${g.id}</span>
      <h3>${g.name}</h3>
      <span class="level-sub">${g.sub}</span>
      <p>${g.text}</p>
    </li>`).join("");

  $("#progression").innerHTML = PROGRESSION.map((p, i) => `
    <div class="prog-item reveal">
      <div class="prog-img">${retinaSVG(i, { label: "Conceptual illustration: " + p.title })}</div>
      <div><b>${p.title}</b><span>${p.text}</span></div>
    </div>`).join("");
}

/* ---------------------------------------------------------------- bar chart helper
   rows: [{label, value (0..1 for width), text (shown at the tip), top (highlight), swatch}] */
function barChart(container, rows) {
  container.innerHTML = rows.map((r) => `
    <div class="bar-row${r.top ? " is-top" : ""}" role="row" style="--g:${r.swatch || "transparent"}">
      <span class="bar-label" role="rowheader">${r.swatch ? "<i></i>" : ""}${r.label}</span>
      <div class="bar-track" title="${r.label}: ${r.text}"><div class="bar-fill" data-w="${(r.value * 100).toFixed(2)}"></div></div>
      <span class="bar-value" role="cell">${r.text}</span>
    </div>`).join("");
  // set widths on the next frame so the CSS transition animates them
  requestAnimationFrame(() => container.querySelectorAll(".bar-fill").forEach((f) => { f.style.width = f.dataset.w + "%"; }));
}

/* ---------------------------------------------------------------- server status + model info */
async function loadHealth() {
  const pill = $("#server-pill"), label = pill.querySelector("span");
  try {
    const h = await (await fetch(`${API}/health`)).json();
    pill.classList.add(h.model_loaded ? "ok" : "bad");
    label.textContent = h.model_loaded ? "Model ready" : "Model not loaded";
  } catch {
    pill.classList.add("bad");
    label.textContent = "Server offline";
  }
}

async function loadModelInfo() {
  const set = (key, text) => { const el = document.querySelector(`[data-info="${key}"]`); if (el) el.textContent = text; };
  try {
    const info = await (await fetch(`${API}/model-info`)).json();
    set("framework", info.framework ? `TensorFlow / Keras (trained with ${info.framework})` : "TensorFlow / Keras");
    if (info.input_size) set("input", `Retinal fundus image → ${info.input_size.join(" × ")}`);
    if (info.preprocessing && info.preprocessing.steps) set("preprocessing", info.preprocessing.steps.join(" · "));
    const t = info.training || {};
    if (t.split_sizes) {
      const s = t.split_sizes;
      set("training", `ImageNet transfer learning · ${t.epochs_trained} epochs · split ${s.train} / ${s.val} / ${s.test}`);
    } else {
      set("training", "Unavailable");
    }
    if (info.upload_limits) $("#limit-mb").textContent = info.upload_limits.max_file_size_mb;
  } catch {
    ["framework", "preprocessing", "training"].forEach((k) => set(k, "Unavailable (server offline)"));
  }
}

/* ---------------------------------------------------------------- analysis (upload -> /predict -> result) */
let selectedFile = null;

function show(state) {   // state: empty | loading | error | body
  ["empty", "loading", "error", "body"].forEach((s) => { $(`#result-${s}`).hidden = s !== state; });
}

function chooseFile(file) {
  if (!file) return;
  selectedFile = file;
  const img = $("#preview");
  img.onerror = () => { img.hidden = true; $("#dz-nopreview").hidden = false; };   // e.g. not an image
  img.src = URL.createObjectURL(file);
  img.hidden = false;
  $("#dz-nopreview").hidden = true;
  $("#dz-empty").hidden = true;
  $("#file-row").hidden = false;
  $("#file-name").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  $("#analyze-btn").disabled = false;
  show("empty");
}

function clearFile() {
  selectedFile = null;
  $("#file-input").value = "";
  $("#preview").hidden = true;
  $("#preview").removeAttribute("src");
  $("#dz-nopreview").hidden = true;
  $("#dz-empty").hidden = false;
  $("#file-row").hidden = true;
  $("#analyze-btn").disabled = true;
  show("empty");
}

function setupUpload() {
  const dz = $("#dropzone"), input = $("#file-input");
  input.addEventListener("change", () => chooseFile(input.files[0]));
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, () => dz.classList.remove("over")));
  dz.addEventListener("drop", (e) => { e.preventDefault(); chooseFile(e.dataTransfer.files[0]); });
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  $("#clear-btn").addEventListener("click", clearFile);
  $("#analyze-btn").addEventListener("click", analyze);
}

async function analyze() {
  if (!selectedFile) return;
  const btn = $("#analyze-btn");
  btn.disabled = true;
  btn.textContent = "Analyzing…";
  show("loading");

  const form = new FormData();
  form.append("file", selectedFile);                 // field name must be "file"
  try {
    const res = await fetch(`${API}/predict`, { method: "POST", body: form });
    const body = await res.json();
    if (!body.success) {                             // every error: {success:false, error:{code,message}}
      showError(body.error.message, `${body.error.code} · HTTP ${res.status}`);
    } else {
      renderResult(body);
    }
  } catch {
    showError("Could not reach the server. Make sure the backend is running.", "NETWORK_ERROR");
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze Image";
  }
}

function showError(message, code) {
  $("#error-message").textContent = message;
  $("#error-code").textContent = code;
  show("error");
}

function renderResult(body) {
  const p = body.prediction;
  const g = GRADES[p.class_id];

  $("#grade-chip").style.setProperty("--g", gradeVar(p.class_id));
  $("#grade-chip").style.setProperty("--gi", gradeInk(p.class_id));
  $("#grade-num").textContent = p.class_id;
  $("#grade-name").textContent = p.class_name;
  $("#grade-desc").textContent = g ? g.text : "";
  $("#conf-value").textContent = pct(p.confidence);   // exactly the value returned by the backend

  barChart($("#prob-chart"), GRADES.map((gr) => {
    const prob = p.probabilities[gr.api] ?? 0;
    return { label: `${gr.id} · ${gr.name}`, value: prob, text: pct(prob), top: gr.id === p.class_id, swatch: gradeVar(gr.id) };
  }));

  $("#result-meta").textContent =
    `${body.filename} · ${body.image_size.width} × ${body.image_size.height} px · ` +
    `processed in ${Math.round(body.processing_time_ms)} ms · ${body.model}`;
  show("body");
  if (window.innerWidth < 900) $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------------------------------------------------------------- research (real results file) */
function renderDistribution() {
  const total = CLASS_COUNTS.reduce((a, b) => a + b, 0), max = Math.max(...CLASS_COUNTS);
  barChart($("#dist-chart"), GRADES.map((g, i) => ({
    label: `${g.id} · ${g.name}`, value: CLASS_COUNTS[i] / max,
    text: CLASS_COUNTS[i].toLocaleString(), swatch: gradeVar(g.id),
  })));
  $("#dist-chart").setAttribute("title", `Total: ${total.toLocaleString()} images`);
}

async function loadResults() {
  let m;
  try {
    m = await (await fetch(`${API}/project-results/densenet121_metrics.json`)).json();
  } catch {
    $("#stat-row").innerHTML = `<p class="muted">Results file unavailable.</p>`;
    $("#compare-body").rows[0].cells[1].textContent = "Unavailable";
    return;
  }

  $("#test-size").textContent = `${m.test_images} test images the model never saw during training`;
  const stats = [
    ["Accuracy", pct(m.accuracy), "share of test images graded correctly"],
    ["Quadratic weighted kappa", m.quadratic_weighted_kappa.toFixed(3), "official APTOS metric · 1.0 = perfect"],
    ["Macro F1", m.macro_f1.toFixed(3), "average over the five grades"],
    ["Training", m.epochs_trained ? `${m.epochs_trained} epochs` : "—", m.train_minutes ? `${m.train_minutes} min on a Colab T4 GPU` : ""],
  ];
  $("#stat-row").innerHTML = stats.map(([k, v, s]) => `<div class="stat"><span>${k}</span><b>${v}</b><small>${s}</small></div>`).join("");

  const report = m.classification_report;
  barChart($("#recall-chart"), GRADES.map((g) => {
    const r = report[g.api];
    return { label: `${g.id} · ${g.name}`, value: r.recall, text: pct(r.recall, 0), swatch: gradeVar(g.id) };
  }));

  // confusion matrix: shade = share of the true grade's images (row-normalised); numbers = image counts
  const cm = m.confusion_matrix;
  let html = `<div class="h"></div>` + GRADES.map((g) => `<div class="h" title="Predicted ${g.name}">${g.id}</div>`).join("");
  cm.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    html += `<div class="rh">${i} · ${GRADES[i].short}</div>`;
    row.forEach((n, j) => {
      const share = n / total;
      const bg = `color-mix(in srgb, var(--accent) ${Math.round(share * 100)}%, var(--track))`;
      const ink = share > 0.7 ? "#fff" : "inherit";   // white only on the darkest cells
      html += `<div class="cell${i === j ? " diag" : ""}" style="background:${bg};color:${ink}" title="True ${GRADES[i].name}, predicted ${GRADES[j].name}: ${n} images (${pct(share, 0)})">${n}</div>`;
    });
  });
  $("#cm").innerHTML = html;

  const params = m.total_params ? `${(m.total_params / 1e6).toFixed(1)} M` : "—";
  $("#compare-body").rows[0].innerHTML = `
    <td>DenseNet121</td><td class="num">${pct(m.accuracy)}</td><td class="num">${m.quadratic_weighted_kappa.toFixed(3)}</td>
    <td class="num">${m.macro_f1.toFixed(3)}</td><td class="num">${params}</td><td><span class="status done">Completed</span></td>`;
}

/* ---------------------------------------------------------------- start */
renderEducation();
renderDistribution();
setupNav();
setupUpload();
setupReveal();
loadHealth();
loadModelInfo();
loadResults();
