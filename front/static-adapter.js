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
    try {
      const respuesta = await fetch("/api/metrics");
      const datos = await respuesta.json();
      extras.innerHTML = `
        <section class="static-view static-metrics" aria-label="Metricas">
          <div class="metrics-head">
            <div>
              <strong>Métricas FiQA</strong>
              <span>${datos.document_count || 0} documentos · ${Object.keys(datos.queries || {}).length} consultas evaluadas</span>
            </div>
            <span class="metrics-status">${datos.evaluation_ready ? "Evaluación completa" : datos.evaluation_loading ? "Calculando..." : "Pendiente"}</span>
          </div>
          ${renderEvaluation(datos)}
        </section>`;
      setupMetricsTabs(extras);
    } catch (error) {
      extras.innerHTML = `<section class="static-view"><p>No se pudieron cargar las métricas.</p></section>`;
    }
  } else {
    if (root) root.dataset.view = "buscar";
    if (results) results.style.removeProperty("display");
    extras.innerHTML = "";
  }
}

function renderEvaluation(datos) {
  if (!datos.evaluation_ready) {
    return `
      <div class="metrics-evaluation metrics-empty">
        ${datos.evaluation_error ? `<p>No se pudo calcular: ${escapeHtml(datos.evaluation_error)}</p>` : "<p>Calculando métricas FIQA…</p>"}
      </div>`;
  }

  const lengthGroups = Object.entries(datos.groups.length || {});
  const typeGroups = Object.entries(datos.groups.type || {});
  const queryEntries = Object.entries(datos.queries || {});

  return `
    <div class="metrics-summary">
      <div class="metrics-section-head">
        <strong>Resumen global</strong>
        <span>648 consultas · MAP</span>
      </div>
      <div class="metrics-map-chart">
        ${Object.entries(datos.summary).map(([model, metrics]) => `
          <div class="metrics-map-row">
            <span>${escapeHtml(model)}</span>
            <div class="metrics-map-track"><i style="width:${Math.min(100, Number(metrics.MAP || 0) * 100)}%"></i></div>
            <strong>${Number(metrics.MAP || 0).toFixed(4)}</strong>
          </div>`).join("")}
      </div>
    </div>

    <div class="metrics-classification">
      <div class="metrics-section-head">
        <strong>Clasificación de consultas</strong>
        <span>Cortas ≤ ${datos.groups.cuts?.short || 0} · Medianas ≤ ${datos.groups.cuts?.medium || 0} tokens</span>
      </div>
      <div class="metrics-tabs" role="tablist">
        <button type="button" class="metrics-tab is-active" data-metrics-tab="length" role="tab" aria-selected="true">Longitud</button>
        <button type="button" class="metrics-tab" data-metrics-tab="type" role="tab" aria-selected="false">Tipo de pregunta</button>
      </div>
      <div class="metrics-tab-panel" data-metrics-panel="length">
        ${lengthGroups.map(([name, ids]) => `
          <div class="metrics-group-line">
            <span class="rank num">•</span>
            <span>${escapeHtml(name)}</span>
            <strong>${ids.length}</strong>
            <small>consultas</small>
          </div>`).join("")}
      </div>
      <div class="metrics-tab-panel" data-metrics-panel="type" hidden>
        ${typeGroups.map(([name, ids]) => `
          <div class="metrics-group-line">
            <span class="rank num">•</span>
            <span>${escapeHtml(name)}</span>
            <strong>${ids.length}</strong>
            <small>consultas</small>
          </div>`).join("")}
      </div>
    </div>

    <div class="metrics-query-analysis">
      <div class="metrics-section-head">
        <strong>Análisis por consulta</strong>
        <span>${queryEntries.length} consultas disponibles</span>
      </div>
      <div class="metrics-query-selector">
        <span>Buscar por índice o texto de la consulta</span>
        <input id="metrics-query-search" type="search" autocomplete="off" placeholder="Ej. 10034 o mutual fund">
        <div class="metrics-query-suggestions" id="metrics-query-suggestions" role="listbox" aria-label="Consultas encontradas"></div>
      </div>
      <div class="metrics-query-detail" id="metrics-query-detail"></div>
    </div>`;
}

function setupMetricsTabs(root) {
  root.querySelectorAll(".metrics-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      root.querySelectorAll(".metrics-tab").forEach((item) => {
        const active = item === tab;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-selected", String(active));
      });
      root.querySelectorAll(".metrics-tab-panel").forEach((panel) => {
        panel.hidden = panel.dataset.metricsPanel !== tab.dataset.metricsTab;
      });
    });
  });

  const search = root.querySelector("#metrics-query-search");
  const suggestions = root.querySelector("#metrics-query-suggestions");
  if (!search || !suggestions) return;
  const detail = root.querySelector("#metrics-query-detail");
  let queryEntries = [];
  const fetchAndRender = async (queryId) => {
    try {
      const response = await fetch("/api/metrics");
      const datos = await response.json();
      queryEntries = Object.entries(datos.queries || {});
      renderQueryDetail(datos, queryId, detail);
    } catch (error) {
      detail.innerHTML = `<p class="metrics-empty-text">No se pudo cargar la consulta.</p>`;
    }
  };

  const showMatches = async (term = "") => {
    try {
      const response = await fetch("/api/metrics");
      const datos = await response.json();
      queryEntries = Object.entries(datos.queries || {});
      const normalized = term.trim().toLowerCase();
      const matches = queryEntries.filter(([qid, query]) =>
        !normalized ||
        qid.toLowerCase().includes(normalized) ||
        query.text.toLowerCase().includes(normalized)
      ).slice(0, 648);
      suggestions.innerHTML = matches.map(([qid, query]) => `
        <button type="button" role="option" data-query-id="${escapeHtml(qid)}">
          <b>${escapeHtml(qid)}</b>
          <span>${escapeHtml(query.text)}</span>
        </button>`).join("");
      suggestions.hidden = matches.length === 0;
      suggestions.querySelectorAll("[data-query-id]").forEach((button) => {
        button.addEventListener("click", () => {
          search.value = `${button.dataset.queryId} — ${button.querySelector("span").textContent}`;
          suggestions.hidden = true;
          fetchAndRender(button.dataset.queryId);
        });
      });
    } catch (error) {
      suggestions.hidden = true;
    }
  };

  search.addEventListener("focus", () => showMatches(search.value));
  search.addEventListener("click", () => showMatches(search.value));
  search.addEventListener("input", () => {
    const term = search.value.trim().toLowerCase();
    if (!term) {
      showMatches("");
      return;
    }
    showMatches(term);
  });
  fetch("/api/metrics")
    .then((response) => response.json())
    .then((datos) => {
      queryEntries = Object.entries(datos.queries || {});
      const firstQueryId = queryEntries[0]?.[0];
      if (firstQueryId) fetchAndRender(firstQueryId);
    })
    .catch(() => {
      detail.innerHTML = `<p class="metrics-empty-text">No se pudo cargar la consulta.</p>`;
    });
}

function renderQueryDetail(datos, queryId, container) {
  const query = datos.queries?.[queryId];
  if (!query) {
    container.innerHTML = `<p class="metrics-empty-text">Consulta no disponible.</p>`;
    return;
  }

  const modelNames = ["Jaccard", "TF-IDF", "BM25"];
  const defaultCuts = [5, 10, 20];
  const cutControls = defaultCuts.map((cut, index) => `
    <label class="metrics-cut-control">
      <span>Corte ${index + 1}</span>
      <input type="number" min="1" max="100" value="${cut}" data-metrics-cut="${index}">
    </label>`).join("");
  const columns = defaultCuts.flatMap((cut) => [`Precision@${cut}`, `Recall@${cut}`]);
  container.innerHTML = `
    <div class="metrics-query-summary">
      <div>
        <span class="metrics-query-id">${escapeHtml(queryId)}</span>
        <p>${escapeHtml(query.text)}</p>
      </div>
      <div class="metrics-query-tags">
        <p class="metrics-procedencia">
          <span class="rank num">•</span>
          <span>${escapeHtml(query.length_group)}</span>
          <span>${escapeHtml(query.query_type)}</span>
          <span>${query.token_count} tokens</span>
        </p>
      </div>
    </div>
    <div class="metrics-table-wrap">
      <div class="metrics-cut-controls">${cutControls}</div>
      <table class="metrics-table query-metrics-table">
        <thead>
          <tr><th>Modelo</th>${columns.map((name) => `<th>${escapeHtml(name)}</th>`).join("")}<th>AP</th></tr>
        </thead>
        <tbody>
          ${modelNames.map((modelName) => {
            const metrics = query.metrics[modelName] || {};
            return `<tr>
              <th scope="row">${escapeHtml(modelName)}</th>
              ${columns.map((name) => `<td>${Number(metrics[name] || 0).toFixed(4)}</td>`).join("")}
              <td>${Number(metrics.AP || 0).toFixed(4)}</td>
            </tr>`;
          }).join("")}
        </tbody>
      </table>
    </div>
    <div class="metrics-cut-status" hidden></div>`;
  container.querySelectorAll("[data-metrics-cut]").forEach((input) => {
    input.addEventListener("change", async () => {
      const inputs = [...container.querySelectorAll("[data-metrics-cut]")]
        .map((item) => Number(item.value));
      const table = container.querySelector(".query-metrics-table");
      const headers = [...table.querySelectorAll("thead th")].slice(1);
      inputs.forEach((cut, cutIndex) => {
        if (headers[cutIndex * 2] && headers[cutIndex * 2 + 1]) {
          headers[cutIndex * 2].textContent = `Precision@${cut}`;
          headers[cutIndex * 2 + 1].textContent = `Recall@${cut}`;
        }
      });

      const valid = inputs.length >= 1
        && inputs.every((cut) => Number.isInteger(cut) && cut > 0)
        && new Set(inputs).size === inputs.length;
      if (!valid) {
        table.querySelectorAll("tbody tr").forEach((row) => {
          inputs.forEach((_, cutIndex) => {
            const precisionCell = row.children[1 + cutIndex * 2];
            const recallCell = row.children[2 + cutIndex * 2];
            if (precisionCell) precisionCell.textContent = "####";
            if (recallCell) recallCell.textContent = "####";
          });
        });
        return;
      }

      try {
        const response = await fetch(`/api/metrics?cuts=${inputs.join(",")}`);
        const dynamic = await response.json();
        if (dynamic.cut_error) return;
        modelNames.forEach((modelName, modelIndex) => {
          const row = table.querySelectorAll("tbody tr")[modelIndex];
          const metrics = dynamic.dynamic_per_query?.[modelName]?.[queryId] || {};
          inputs.forEach((cut, cutIndex) => {
            const precisionCell = row?.children[1 + cutIndex * 2];
            const recallCell = row?.children[2 + cutIndex * 2];
            if (precisionCell) precisionCell.textContent = Number(metrics[`Precision@${cut}`] || 0).toFixed(4);
            if (recallCell) recallCell.textContent = Number(metrics[`Recall@${cut}`] || 0).toFixed(4);
          });
        });
      } catch (error) {
        table.querySelectorAll("tbody tr").forEach((row) => {
          inputs.forEach((_, cutIndex) => {
            const precisionCell = row.children[1 + cutIndex * 2];
            const recallCell = row.children[2 + cutIndex * 2];
            if (precisionCell) precisionCell.textContent = "####";
            if (recallCell) recallCell.textContent = "####";
          });
        });
      }
    });
  });
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
