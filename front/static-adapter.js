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
      if (datos.error) {
        extras.innerHTML = `<section class="static-view"><p>${datos.error}</p></section>`;
        return;
      }
      const nombres = { jaccard: "Jaccard", tfidf: "TF-IDF", bm25: "BM25" };
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
          <h2>Estado del corpus</h2>
          <div class="static-metric-cards">
            <article><strong>Documentos</strong><span>${datos.document_count || 0}</span></article>
            <article><strong>Términos TF-IDF</strong><span>${datos.term_count || 0}</span></article>
            <article><strong>Estado</strong><span>${datos.available ? "Listo" : "Pendiente"}</span></article>
          </div>
          <p>${escapeHtml(datos.source || "No hay corpus importado.")}</p>
          <form id="corpus-upload">
            <input type="file" name="file" accept=".json,.jsonl,.txt">
            <button type="submit">Importar corpus</button>
          </form>
        </section>`;
      document.querySelector("#corpus-upload")?.addEventListener("submit", importarArchivo);
    } catch (error) {
      extras.innerHTML = `<section class="static-view"><p>No se pudieron cargar las métricas.</p></section>`;
    }
  } else {
    if (root) root.dataset.view = "buscar";
    if (results) results.style.removeProperty("display");
    extras.innerHTML = "";
  }
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

async function importarArchivo(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const file = form.querySelector("input")?.files?.[0];
  if (!file) return;
  const payload = new FormData();
  payload.append("file", file);
  await fetch("/api/corpus/import", { method: "POST", body: payload });
  form.innerHTML = "<p>Importando corpus...</p>";
  await esperarCorpus();
  mostrarExtras("metricas");
}

async function esperarCorpus() {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const respuesta = await fetch("/api/corpus/status");
    const estadoCorpus = await respuesta.json();
    if (estadoCorpus.ready || estadoCorpus.error) return estadoCorpus;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  return { ready: false, error: "Tiempo de espera agotado" };
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
