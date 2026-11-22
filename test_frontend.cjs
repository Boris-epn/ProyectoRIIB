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
