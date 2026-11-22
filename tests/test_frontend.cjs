const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync('front/static-adapter.js', 'utf8').replace(/iniciar\(\);\s*$/, '');
const context = vm.createContext({document: {documentElement: {dataset: {}}}});
vm.runInContext(source, context);
const data = {
  evaluation_ready: true, dynamic_cuts: [3, 15],
  summary: Object.fromEntries(['Jaccard', 'TF-IDF', 'BM25', 'Embeddings'].map(name => [name, {
    MAP: .5, Precision: .2, Recall: .8, 'Precision@3': 1 / 3, 'Recall@3': .5,
  }])),
  groups: {cuts: {}, length: {}, type: {}},
  queries: {q1: {text: '<script>unsafe</script>', metrics: {Embeddings: {AP: .75, 'Precision@3': .333}}}},
};
context.data = data;
const html = vm.runInContext('renderEvaluation(data)', context);
for (const name of ['Jaccard', 'TF-IDF', 'BM25', 'Embeddings', 'MAP', 'Precision', 'Recall']) assert(html.includes(name));
assert(!html.includes('Precision@3'));
assert(!html.includes('Recall@15'));
assert(!html.includes('648'));
const container = {};
context.container = container;
vm.runInContext('renderQueryDetail(data, "q1", container)', context);
assert(container.innerHTML.includes('Embeddings'));
assert(container.innerHTML.includes('Precision@3'));
assert(container.innerHTML.includes('Recall@15'));
assert(container.innerHTML.includes('0.7500'));
assert(container.innerHTML.includes('&lt;script&gt;'));
assert(!container.innerHTML.includes('<script>'));
assert.equal(vm.runInContext('comparisonScore("dense", {score: -0.25})', context), 'Coseno -0.2500');
console.log('Frontend: cortes personalizados, cuatro modelos, AP, escape HTML y coseno: OK');

delete data.summary.Embeddings;
data.queries.q1.metrics = {};
data.dense_evaluation_loading = true;
data.dense_evaluation_progress = {done: 1, total: 2};
data.dense = {state: 'ready'};
const pending = vm.runInContext('renderEvaluation(data)', context);
assert(pending.includes('<th scope="row">Embeddings</th>'));
assert(pending.includes('Evaluando consultas: 1/2'));
vm.runInContext('renderQueryDetail(data, "q1", container)', context);
assert(container.innerHTML.includes('<th scope="row">Embeddings</th>'));
assert(container.innerHTML.includes('Evaluando consultas: 1/2'));
assert(!container.innerHTML.includes('0.0000'));
data.dense_evaluation_error = '<fallo del modelo>';
const failed = vm.runInContext('renderEvaluation(data)', context);
assert(failed.includes('Error: &lt;fallo del modelo&gt;'));
console.log('Embeddings visible en ambas tablas durante carga y error: OK');

// Casos empíricos y agrupación con resultados calculados por el servidor.
data.analysis = {
  completed_models: ['Jaccard', 'TF-IDF', 'BM25', 'Embeddings'],
  by_group: {type: [{group: 'definicion', count: 2, scores: {Embeddings: .7, BM25: .4}, winners: ['Embeddings']}]},
  best_types: {Embeddings: {groups: ['definicion'], MAP: .7}},
  semantic: {ready: true, comparisons: {BM25: {
    count: 2, wins: 1, losses: 0, ties: 1, mean_delta_ap: .25,
    improvements: [{query_id: 'q1', text: '<script>unsafe</script>', query_type: 'definicion',
      dense_ap: .8, baseline_ap: .3, delta_ap: .5, first_relevant_dense: 1, first_relevant_baseline: null}],
    regressions: [],
  }}},
};
const groupHtml = vm.runInContext('renderGroupPerformance(data)', context);
assert(groupHtml.includes('MAP Embeddings'));
assert(groupHtml.includes('0.7000'));
assert(groupHtml.includes('Definición'));
const casesHtml = vm.runInContext('renderSemanticComparison(data)', context);
assert(casesHtml.includes('+0.5000'));
assert(casesHtml.includes('data-analysis-query="q1"'));
assert(casesHtml.includes('&lt;script&gt;'));
assert(!casesHtml.includes('<script>'));
assert(casesHtml.includes('No se encontraron casos'));
assert(casesHtml.includes('no recuperado'));
vm.runInContext('referenciaSemantica = "TF-IDF"', context);
const unavailable = vm.runInContext('renderSemanticComparison(data)', context);
assert(!unavailable.includes('data-analysis-query'));
console.log('Análisis por tipo y casos semánticos, empate, datos pendientes y escape HTML: OK');
