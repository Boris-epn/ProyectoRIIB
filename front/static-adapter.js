const sugerencias = [
  "What is a mutual fund?",
  "How do I build wealth?",
  "Is a car loan bad debt?",
  "How to invest in index funds?",
];
const vocabulario = [
  "account", "budget", "credit", "debt", "investment", "mortgage",
  "retirement", "savings", "stocks", "tax",
];

let estado = {
  q: "",
  model: "tfidf",
  k: 10,
  stem: false,
  graph: false,
  view: "buscar",
  route: "home",
  seq: 0,
  nav: 0,
};
let recibir;

let reintentoBusqueda = null;
let reintentoComparar = null;
let reintentoMetricas = null;
let cortesMetricas = [5, 10, 20];
let consultaMetrica = null;
let metricasVersion = 0;

const DENSE_PENDIENTE = new Set(["idle", "loading", "building"]);

document.documentElement.dataset.theme = "dark";
document.title = "goggle";

async function buscar(solicitud) {
  const respuesta = await fetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(solicitud),
  });
  if (!respuesta.ok) throw new Error(`Error HTTP ${respuesta.status}`);
  return respuesta.json();
}

async function establecerSolicitud(solicitud) {
  estado = { ...estado, ...solicitud };
  try {
    const response = await buscar(estado);
    recibir?.({
      initial: { ...estado, nav: estado.nav + 1 },
      response,
    });
    if (DENSE_PENDIENTE.has(response.dense?.state) && estado.model === "dense") {
      clearTimeout(reintentoBusqueda);
      reintentoBusqueda = setTimeout(() => establecerSolicitud({ ...estado }), 2000);
    }
  } catch (error) {
    recibir?.({
      initial: { ...estado, nav: estado.nav + 1 },
      response: {
        seq: estado.seq,
        model: estado.model,
        elapsed_ms: 0,
        terms: [],
        suggestion: "",
        hits: [],
        error: error.message,
      },
    });
  }
  mostrarExtras(estado.view);
}

async function mostrarExtras(vista) {
  clearTimeout(reintentoMetricas);
  if (vista !== "comparar") clearTimeout(reintentoComparar);
  if (vista !== "metricas") metricasVersion += 1;
  const extras = document.querySelector("#demo-extras");
  const root = document.querySelector(".goggle");
  const results = root?.querySelector(".results");
  const main = root?.querySelector(".goggle-main");
  if (vista === "comparar") {
    if (root) root.dataset.view = "comparar";
    if (results) results.style.display = "none";
    if (main && !main.contains(extras)) main.append(extras);
    extras.innerHTML = `<section class="static-view"><p>Cargando comparacion...</p></section>`;
    if (!estado.q.trim()) {
      extras.innerHTML = "";
      return;
    }
    try {
      const respuesta = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: estado.q, k: estado.k }),
      });
      const datos = await respuesta.json();
      if (DENSE_PENDIENTE.has(datos.dense?.state)) {
        clearTimeout(reintentoComparar);
        reintentoComparar = setTimeout(() => mostrarExtras("comparar"), 2000);
      }
      if (datos.error) {
        extras.innerHTML = `<section class="static-view"><p>${datos.error}</p></section>`;
        return;
      }
      const nombres = { jaccard: "Jaccard", tfidf: "TF-IDF", bm25: "BM25", dense: "Embeddings" };
      extras.innerHTML = `
        <section class="static-view static-compare" aria-label="Comparacion">
          <div class="static-controls">
            <strong>Comparación Top ${datos.k}</strong>
            <p>Referencias: Top ${datos.k} de BM25</p>
            <button type="button" class="compare-back">Regresar a resultados</button>
          </div>
          <div class="static-compare-columns">
            ${Object.entries(datos.results).map(([modelo, hits]) => `
              <section class="static-column">
                <header class="static-column-header">
                  <div>
                    <strong>${nombres[modelo]}</strong>
                  </div>
                  <div class="static-header-metrics">
                    <span><small>Precision@${datos.k}</small><b>${datos.metrics[modelo].precision.toFixed(4)}</b></span>
                    <span><small>Recall@${datos.k}</small><b>${datos.metrics[modelo].recall.toFixed(4)}</b></span>
                  </div>
                </header>
                ${hits.map((hit) => `
                  <article class="static-document goggle">
                    <div class="row static-row">
                      <div>
                        <p class="procedencia">
                          <span class="rank num">${hit.rank}</span>
                          <span>Doc. ${escapeHtml(hit.doc_id)}</span>
                          <span class="relevancia">${comparisonScore(modelo, hit)}</span>
                        </p>
                        <p class="lead static-lead">${escapeHtml(hit.text.slice(0, 110))}${hit.text.length > 110 ? "…" : ""}</p>
                        ${hit.text.length > 110 ? `<p class="snippet static-snippet">${escapeHtml(hit.text.slice(110, 220))}${hit.text.length > 220 ? "…" : ""}</p>` : ""}
                        ${hit.text.length > 110 ? `<button class="more static-more" type="button" data-full="${escapeHtml(hit.text)}" aria-haspopup="dialog">Ver más</button>` : ""}
                      </div>
                    </div>
                  </article>`).join("") || "<div class=\"static-document\"><p>Sin resultados</p></div>"}
              </section>`).join("")}
          </div>
        </section>
        <dialog class="document-dialog" aria-labelledby="document-dialog-title">
          <div class="document-dialog-head">
            <strong id="document-dialog-title">Documento completo</strong>
            <button type="button" class="document-dialog-close" aria-label="Cerrar"><span aria-hidden="true">×</span></button>
          </div>
          <p class="document-dialog-text"></p>
        </dialog>`;
      extras.querySelectorAll(".static-more").forEach((button) => {
        button.addEventListener("click", () => {
          const dialog = extras.querySelector(".document-dialog");
          const text = button.dataset.full || "";
          dialog.querySelector(".document-dialog-text").textContent = text;
          if (typeof dialog.showModal === "function") dialog.showModal();
          else dialog.setAttribute("open", "");
        });
      });
      extras.querySelector(".document-dialog-close")?.addEventListener("click", () => {
        extras.querySelector(".document-dialog")?.close();
      });
      extras.querySelector(".compare-back")?.addEventListener("click", () => {
        document.dispatchEvent(new CustomEvent("goggle:compare"));
      });
    } catch (error) {
      extras.innerHTML = `<section class="static-view"><p>No se pudo comparar: ${escapeHtml(error.message)}</p></section>`;
    }
  } else if (vista === "metricas") {
    if (root) root.dataset.view = "metricas";
    if (results) results.style.display = "none";
    if (main && !main.contains(extras)) main.append(extras);
    await cargarMetricas(extras);
  } else {
    if (root) root.dataset.view = "buscar";
    if (results) results.style.removeProperty("display");
    extras.innerHTML = "";
  }
}

async function cargarMetricas(extras) {
  const version = ++metricasVersion;
  try {
    const response = await fetch(`/api/metrics?cuts=${cortesMetricas.join(",")}`);
    const datos = await response.json();
    if (version !== metricasVersion || estado.view !== "metricas") return;
    if (!response.ok) throw new Error(datos.cut_error || datos.error);
    const count = Object.keys(datos.queries || {}).length;
    const pending = datos.evaluation_loading || datos.dense_evaluation_loading;
    extras.innerHTML = `
      <section class="static-view static-metrics" aria-label="Evaluación de los modelos">
        <div class="metrics-head"><div><strong>Evaluación de los modelos</strong>
          <span>${datos.document_count || 0} documentos · ${count} consultas evaluadas</span></div>
          <span class="metrics-status">${pending ? "Calculando..." : datos.evaluation_ready ? "Evaluación disponible" : "Pendiente"}</span>
        </div>
        <form class="metrics-cut-form">
          <label for="metrics-cuts">Cortes para Precision@k y Recall@k por consulta</label>
          <div><input id="metrics-cuts" type="text" value="${cortesMetricas.join(", ")}" aria-describedby="metrics-cuts-help" required>
          <button type="submit">Aplicar cortes</button></div>
          <small id="metrics-cuts-help">Elige de uno a tres enteros positivos distintos, separados por comas. Ejemplo: 3, 15, 50.</small>
          <p class="metrics-cut-error" role="alert" hidden></p>
        </form>
        ${renderEvaluation(datos)}
      </section>`;
    extras.querySelector(".metrics-cut-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const values = extras.querySelector("#metrics-cuts").value.split(",").map(value => value.trim());
      const cuts = values.map(Number);
      const error = extras.querySelector(".metrics-cut-error");
      if (cuts.length < 1 || cuts.length > 3 || values.some(value => !/^\d+$/.test(value))
          || cuts.some(cut => !Number.isSafeInteger(cut) || cut < 1) || new Set(cuts).size !== cuts.length) {
        error.textContent = "Ingresa de uno a tres enteros positivos y distintos.";
        error.hidden = false;
        return;
      }
      cortesMetricas = cuts;
      clearTimeout(reintentoMetricas);
      cargarMetricas(extras);
    });
    setupMetricsTabs(extras, datos);
    if (pending || (datos.available && !datos.evaluation_ready && !datos.evaluation_error)) {
      const refresh = () => {
        // No interrumpir al usuario mientras escribe los cortes o una consulta.
        if (extras.contains(document.activeElement) && document.activeElement?.tagName === "INPUT") {
          reintentoMetricas = setTimeout(refresh, 2500);
        } else {
          cargarMetricas(extras);
        }
      };
      reintentoMetricas = setTimeout(refresh, 2500);
    }
  } catch (error) {
    if (version === metricasVersion) extras.innerHTML = `<section class="static-view"><p>No se pudieron cargar las métricas: ${escapeHtml(error.message)}</p></section>`;
  }
}

function metricsTable(rows, cuts, aggregate = false) {
  const columns = aggregate ? ["MAP", "Precision", "Recall"] : ["AP",
    ...cuts.flatMap(cut => [`Precision@${cut}`, `Recall@${cut}`])];
  return `<div class="metrics-table-wrap"><table class="metrics-table">
    <thead><tr><th>Modelo</th>${columns.map(name => `<th>${name}</th>`).join("")}</tr></thead>
    <tbody>${Object.entries(rows).map(([name, metrics]) => `<tr><th scope="row">${escapeHtml(name)}</th>
      ${columns.map(column => `<td>${Number(metrics[column] || 0).toFixed(4)}</td>`).join("")}</tr>`).join("")}</tbody>
  </table></div>`;
}

function renderEvaluation(datos) {
  if (!datos.evaluation_ready) {
    return `<div class="metrics-evaluation metrics-empty"><p>${datos.evaluation_error
      ? `No se pudo calcular: ${escapeHtml(datos.evaluation_error)}`
      : datos.available ? "Calculando métricas FiQA..." : "Importa el corpus FiQA para evaluar los modelos."}</p></div>`;
  }
  const cuts = datos.dynamic_cuts;
  const count = Object.keys(datos.queries || {}).length;
  const denseNotice = datos.dense_evaluation_error
    ? `No se pudo evaluar Embeddings: ${escapeHtml(datos.dense_evaluation_error)}`
    : datos.dense_evaluation_loading ? `Embeddings: ${escapeHtml(datos.dense?.message || "evaluando consultas...")}` : "";
  return `
    <div class="metrics-summary">
      <div class="metrics-section-head"><strong>Resumen global</strong><span>${count} consultas</span></div>
      <p>Evaluación con los juicios de relevancia oficiales de FiQA. MAP, Precision y Recall son promedios por consulta.</p>
      <p>Precision y Recall generales usan todo el ranking recuperado; las métricas @k del análisis por consulta usan los cortes elegidos. AP y MAP usan todo el ranking y no dependen de esos cortes.</p>
      ${denseNotice ? `<p class="metrics-dense-status" role="status">${denseNotice}</p>` : ""}
      ${metricsTable(datos.summary, cuts, true)}
    </div>
    <div class="metrics-classification">
      <div class="metrics-section-head"><strong>Clasificación de consultas</strong>
        <span>Cortas ≤ ${datos.groups.cuts?.short || 0} · Medianas ≤ ${datos.groups.cuts?.medium || 0} tokens</span></div>
      <div class="metrics-tabs" role="tablist">
        <button type="button" class="metrics-tab is-active" data-metrics-tab="length" role="tab" aria-selected="true">Longitud</button>
        <button type="button" class="metrics-tab" data-metrics-tab="type" role="tab" aria-selected="false">Tipo de pregunta</button>
      </div>
      ${["length", "type"].map(group => `<div class="metrics-tab-panel" data-metrics-panel="${group}" ${group === "type" ? "hidden" : ""}>
        ${Object.entries(datos.groups[group] || {}).map(([name, ids]) => `<div class="metrics-group-line"><span>${escapeHtml(name)}</span><strong>${ids.length}</strong><small>consultas</small></div>`).join("")}
      </div>`).join("")}
    </div>
    <div class="metrics-query-analysis">
      <div class="metrics-section-head"><strong>Análisis por consulta</strong><span>${count} consultas disponibles</span></div>
      <div class="metrics-query-selector"><label for="metrics-query-search">Buscar por índice o texto de la consulta</label>
        <input id="metrics-query-search" type="search" autocomplete="off" placeholder="Ej. 10034 o mutual fund">
        <div class="metrics-query-suggestions" id="metrics-query-suggestions" role="listbox" aria-label="Consultas encontradas" hidden></div>
      </div><div class="metrics-query-detail" id="metrics-query-detail"></div>
    </div>`;
}

function setupMetricsTabs(root, datos) {
  root.querySelectorAll(".metrics-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      root.querySelectorAll(".metrics-tab").forEach(item => {
        const active = item === tab;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-selected", String(active));
      });
      root.querySelectorAll(".metrics-tab-panel").forEach(panel => {
        panel.hidden = panel.dataset.metricsPanel !== tab.dataset.metricsTab;
      });
    });
  });
  const search = root.querySelector("#metrics-query-search");
  if (!search) return;
  const suggestions = root.querySelector("#metrics-query-suggestions");
  const detail = root.querySelector("#metrics-query-detail");
  const entries = Object.entries(datos.queries || {});
  const select = qid => {
    consultaMetrica = qid;
    search.value = `${qid} - ${datos.queries[qid].text}`;
    suggestions.hidden = true;
    renderQueryDetail(datos, qid, detail);
  };
  const showMatches = () => {
    const term = search.value.trim().toLowerCase();
    const matches = entries.filter(([qid, query]) => !term || qid.toLowerCase().includes(term) || query.text.toLowerCase().includes(term));
    suggestions.innerHTML = matches.map(([qid, query]) => `<button type="button" role="option" data-query-id="${escapeHtml(qid)}"><b>${escapeHtml(qid)}</b><span>${escapeHtml(query.text)}</span></button>`).join("");
    suggestions.hidden = matches.length === 0;
    suggestions.querySelectorAll("[data-query-id]").forEach(button => button.addEventListener("click", () => select(button.dataset.queryId)));
  };
  search.addEventListener("input", showMatches);
  search.addEventListener("focus", () => { search.select(); showMatches(); });
  const initial = datos.queries[consultaMetrica] ? consultaMetrica : entries[0]?.[0];
  if (initial) select(initial);
}

function renderQueryDetail(datos, queryId, container) {
  const query = datos.queries?.[queryId];
  if (!query) return;
  container.innerHTML = `<div class="metrics-query-summary"><div><span class="metrics-query-id">${escapeHtml(queryId)}</span>
    <p>${escapeHtml(query.text)}</p></div><div class="metrics-query-tags"><p>${escapeHtml(query.length_group)} · ${escapeHtml(query.query_type)} · ${query.token_count} tokens</p></div></div>
    ${metricsTable(query.metrics, datos.dynamic_cuts)}`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
}

function comparisonScore(model, hit) {
  if (model === "bm25") return hit.score.toFixed(2);
  if (model === "dense") return `Coseno ${hit.score.toFixed(4)}`;
  return `${(hit.score * 100).toFixed(1).replace(".", ",")} %`;
}

async function importarCorpusInicial() {
  const respuesta = await fetch("/api/corpus/status");
  const estadoCorpus = await respuesta.json();
  if (!estadoCorpus.ready && !estadoCorpus.loading && !estadoCorpus.error) {
    await fetch("/api/corpus/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset: "BeIR/fiqa", split: "corpus" }),
    });
  }
}

async function iniciar() {
  const respuesta = await fetch(`app/frontend/app.js?v=${Date.now()}`);
  let codigoFuente = await respuesta.text();
  codigoFuente = codigoFuente.replace("__SUGGESTIONS__", JSON.stringify(sugerencias));
  codigoFuente = codigoFuente.replace("__VOCABULARY__", JSON.stringify(vocabulario));

  const urlMódulo = URL.createObjectURL(new Blob([codigoFuente], { type: "text/javascript" }));
  const módulo = await import(urlMódulo);
  URL.revokeObjectURL(urlMódulo);

  await módulo.default({
    parentElement: document,
    data: {
      initial: estado,
      response: {
        seq: 0,
        key: "",
        model: estado.model,
        elapsed_ms: 0,
        terms: [],
        suggestion: "",
        hits: [],
      },
    },
    setStateValue(_nombre, solicitud) {
      establecerSolicitud(solicitud);
    },
  });

  const root = document.querySelector(".goggle");
  recibir = root?._goggle?.receive;

  document.addEventListener("goggle:compare", () => {
    const volver = estado.view === "comparar";
    estado.view = volver ? "buscar" : "comparar";
    const root = document.querySelector(".goggle");
    const results = root?.querySelector(".results");
    root?.setAttribute("data-view", estado.view);
    if (estado.view === "comparar") {
      if (results) results.style.display = "none";
    } else if (results) {
      results.style.removeProperty("display");
    }
    estado.seq += 1;
    establecerSolicitud({ ...estado, view: estado.view });
  });
  document.querySelectorAll("[data-view-link]").forEach((link) => {
    link.addEventListener("click", () => {
      const view = link.dataset.viewLink;
      estado.view = view;
      estado.seq += 1;
      mostrarExtras(view);
    });
  });
  importarCorpusInicial();
}

iniciar();
