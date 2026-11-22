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

// Mientras el índice de Embeddings se prepara, se repite la misma petición.
const ESPERA_REINTENTO = 2000;
const DENSE_PENDIENTE = new Set(["idle", "loading", "building"]);
const densePendiente = (datos) => DENSE_PENDIENTE.has(datos?.dense?.state);
let reintentoBusqueda = null;
let reintentoComparar = null;
let turnoComparar = 0;

function programarReintentoBusqueda(response) {
  clearTimeout(reintentoBusqueda);
  if (!densePendiente(response)) return;
  const seq = estado.seq;
  reintentoBusqueda = setTimeout(async () => {
    if (estado.seq !== seq) return;  // el usuario cambió de consulta o de modelo
    try {
      const respuesta = await buscar(estado);
      if (estado.seq !== seq) return;
      // Mismo seq y sin subir nav: el front lo trata como respuesta a la búsqueda actual.
      recibir?.({ initial: { ...estado }, response: respuesta });
      programarReintentoBusqueda(respuesta);
    } catch {
      // Un fallo de red detiene los reintentos; la próxima búsqueda lo mostrará.
    }
  }, ESPERA_REINTENTO);
}

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
  clearTimeout(reintentoBusqueda);
  estado = { ...estado, ...solicitud };
  try {
    const response = await buscar(estado);
    recibir?.({
      initial: { ...estado, nav: estado.nav + 1 },
      response,
    });
    if (response.seq === estado.seq) programarReintentoBusqueda(response);
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

async function mostrarExtras(vista, reintento = false) {
  clearTimeout(reintentoComparar);
  const turno = ++turnoComparar;
  const extras = document.querySelector("#demo-extras");
  if (vista === "comparar") {
    if (!reintento) extras.innerHTML = `<section class="static-view"><p>Cargando comparacion...</p></section>`;
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
      if (reintento && turno !== turnoComparar) return;  // llegó tarde: ya hay otra comparación
      if (datos.error) {
        extras.innerHTML = `<section class="static-view"><p>${datos.error}</p></section>`;
        return;
      }
      if (densePendiente(datos)) {
        const firma = JSON.stringify([estado.q, estado.k]);
        reintentoComparar = setTimeout(() => {
          if (estado.view === "comparar" && JSON.stringify([estado.q, estado.k]) === firma) {
            mostrarExtras("comparar", true);
          }
        }, ESPERA_REINTENTO);
      }
      const nombres = { jaccard: "Jaccard", tfidf: "TF-IDF", bm25: "BM25", dense: "Embeddings" };
      extras.innerHTML = `
        <section class="static-view static-compare" aria-label="Comparacion">
          <div class="static-controls">
            <strong>Top ${estado.k}</strong>
            <div><button class="static-selected" type="button">Todos</button></div>
          </div>
          <div class="static-columns">
            ${Object.entries(datos.results).map(([modelo, hits]) => `
              <article class="static-column">
                <header><strong>${nombres[modelo]}</strong><span>${hits.length} resultados</span></header>
                ${hits.map((hit) => `
                  <div class="static-document">
                    <b>${hit.rank}</b>
                    <div><strong>Documento ${escapeHtml(hit.doc_id)}</strong>
                    <p>${escapeHtml(hit.text.slice(0, 180))}${hit.text.length > 180 ? "…" : ""}</p>
                    <small>${hit.score.toFixed(4)}</small></div>
                  </div>`).join("") || `<div class="static-document"><p>${modelo === "dense" && densePendiente(datos) ? escapeHtml(datos.dense.message) : "Sin resultados"}</p></div>`}
              </article>`).join("")}
          </div>
        </section>`;
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
  const respuesta = await fetch("app/frontend/app.js");
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
