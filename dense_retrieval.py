"""Recuperación semántica (modelo "dense") con sentence-transformers y FAISS.

Cada documento (título + texto) se codifica con all-MiniLM-L6-v2 en un vector
de 384 dimensiones normalizado a norma 1. Los vectores van a un
``faiss.IndexFlatIP``: con vectores normalizados, el producto interno es la
similitud coseno, así que los scores quedan en [-1, 1] y la búsqueda es exacta.

Uso como script (genera la caché que luego lee el servidor):

    .\\.venv\\Scripts\\python dense_retrieval.py --build
    .\\.venv\\Scripts\\python dense_retrieval.py --build --limit 200

La caché se guarda en ``.cache/dense/<huella>/`` y la huella depende del modelo
y del contenido y orden exactos de los documentos, de modo que una caché nunca
se reutiliza con otro corpus. ``sentence_transformers``, ``torch`` y ``faiss``
se importan solo cuando hacen falta (carga perezosa).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = ROOT / ".cache" / "dense"
DEFAULT_BATCH_SIZE = 64
CACHE_VERSION = 1
# Documentos por llamada al codificador: marca el ritmo de la barra de progreso
# y deja que sentence-transformers agrupe por longitud dentro de cada bloque.
ENCODE_CHUNK = 1024


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Devuelve una matriz (len(texts), dim) de vectores."""


class SentenceTransformerEncoder:
    """Envoltorio perezoso: el modelo se descarga y carga en el primer uso."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    # Usar primero los archivos descargados sin consultar la red.
                    # No cambiar HF_HUB_OFFLINE: el corpus puede necesitar conexión.
                    try:
                        self._model = SentenceTransformer(
                            self.model_name, device=self.device, local_files_only=True
                        )
                    except OSError:
                        # Primera ejecución: descargar solo si falta la copia local.
                        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.load().encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def normalize_documents(documents: Iterable[dict]) -> list[dict]:
    """Limpia los documentos igual que ``SearchEngine.import_documents``.

    Debe producir los mismos ``_id``, ``title`` y ``text`` que server.py para
    que la huella de la caché coincida. Es idempotente: aplicarla a
    ``engine.documents`` (ya limpios) no cambia nada.
    """
    cleaned = []
    for position, document in enumerate(documents, start=1):
        doc_id = str(document.get("_id") or document.get("id") or position)
        title = str(document.get("title") or "")
        text = str(document.get("text") or document.get("content") or "")
        cleaned.append({"_id": doc_id, "title": title, "text": text})
    return cleaned


def document_text(document: dict) -> str:
    """Texto que se codifica: título + texto, igual que muestra format_hits."""
    return " ".join(value for value in (document["title"], document["text"]) if value)


def corpus_fingerprint(documents: Sequence[dict], model_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"v{CACHE_VERSION}\0{model_name}\0{len(documents)}\0".encode("utf-8"))
    for document in documents:
        for value in (document["_id"], document["title"], document["text"]):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _normalize(vectors) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


class _Cancelled(Exception):
    """El corpus cambió mientras se construía el índice."""


class DenseRetriever:
    """Índice FAISS sobre embeddings de documentos.

    Estados: ``idle`` → ``loading`` (leyendo caché / cargando modelo) →
    ``building`` (codificando el corpus, solo si no hay caché) → ``ready``;
    o ``error``. Al recibir otra lista de documentos (comparada por identidad,
    como la que deja ``import_documents`` al reimportar) vuelve a ``idle``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        encoder: Encoder | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        build_if_missing: bool = True,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.encoder = encoder if encoder is not None else SentenceTransformerEncoder(
            model_name, batch_size
        )
        self.build_if_missing = build_if_missing
        self._state_lock = threading.RLock()
        self._encode_lock = threading.Lock()
        self._documents: list[dict] | None = None
        self._generation = 0
        self._index = None
        self._ids: list[str] = []
        self.state = "idle"
        self.done = 0
        self.total = 0
        self.error = ""
        self.origin = ""
        self.cache_location: Path | None = None

    # ----------------------------------------------------------------- estado
    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def status(self) -> dict:
        with self._state_lock:
            return {
                "state": self.state,
                "done": self.done,
                "total": self.total,
                "error": self.error,
                "origin": self.origin,
                "model": self.model_name,
            }

    def reset(self, documents: list[dict] | None = None) -> None:
        with self._state_lock:
            self._generation += 1
            self._documents = documents
            self._index = None
            self._ids = []
            self.state = "idle"
            self.done = 0
            self.total = len(documents) if documents is not None else 0
            self.error = ""
            self.origin = ""
            self.cache_location = None

    def cache_path(self, fingerprint: str) -> Path:
        return self.cache_dir / fingerprint[:16]

    # ------------------------------------------------------------ preparación
    def ensure_ready(self, documents: list[dict]) -> bool:
        """No bloquea: devuelve True si ya se puede buscar sobre ``documents``.

        Si no, lanza (una sola vez) un hilo que lee la caché o, como respaldo,
        construye el índice. Un estado ``error`` se mantiene hasta ``reset``
        o hasta que cambie el corpus, para no reintentar en cada petición.
        """
        with self._state_lock:
            if documents is not self._documents:
                self.reset(documents)
            if self.state == "ready":
                return True
            if self.state == "idle":
                self.state = "loading"
                generation = self._generation
                threading.Thread(
                    target=self._prepare_in_background,
                    args=(documents, generation),
                    daemon=True,
                ).start()
            return False

    def build(self, documents: list[dict], force: bool = False, show_progress: bool = False) -> dict:
        """Prepara el índice de forma bloqueante (script y pruebas).

        Usa la caché si es válida; si no (o con ``force``), codifica el corpus
        y la guarda.
        """
        with self._state_lock:
            if documents is not self._documents:
                self.reset(documents)
            self.state = "loading"
            generation = self._generation
        try:
            self._load_or_build(documents, generation, allow_build=True,
                                force=force, show_progress=show_progress)
        except Exception as exc:
            self._fail(generation, exc)
            raise
        return self.status()

    def _prepare_in_background(self, documents: list[dict], generation: int) -> None:
        try:
            self._load_or_build(documents, generation, allow_build=self.build_if_missing)
        except _Cancelled:
            pass
        except Exception as exc:  # se informa a través de status()
            self._fail(generation, exc)

    def _fail(self, generation: int, exc: Exception) -> None:
        with self._state_lock:
            if generation == self._generation:
                self.state = "error"
                self.error = str(exc) or exc.__class__.__name__

    def _load_or_build(
        self,
        documents: list[dict],
        generation: int,
        allow_build: bool,
        force: bool = False,
        show_progress: bool = False,
    ) -> None:
        cleaned = normalize_documents(documents)
        if not cleaned:
            raise ValueError("El corpus está vacío")
        fingerprint = corpus_fingerprint(cleaned, self.model_name)
        ids = [document["_id"] for document in cleaned]
        path = self.cache_path(fingerprint)

        embeddings = None if force else self._read_cache(path, fingerprint, ids)
        origin = "cache"
        if embeddings is None:
            if not allow_build:
                raise RuntimeError(
                    "No hay caché de embeddings para este corpus. "
                    "Genérala con: python dense_retrieval.py --build"
                )
            embeddings = self._encode_corpus(cleaned, generation, show_progress)
            self._write_cache(path, fingerprint, ids, embeddings)
            origin = "built"

        import faiss

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # Deja el modelo cargado para que la primera consulta no espere.
        load = getattr(self.encoder, "load", None)
        if callable(load):
            load()

        with self._state_lock:
            if generation != self._generation:
                raise _Cancelled()
            self._index = index
            self._ids = ids
            self.state = "ready"
            self.origin = origin
            self.done = self.total = len(ids)
            self.cache_location = path

    def _encode_corpus(self, cleaned: list[dict], generation: int, show_progress: bool) -> np.ndarray:
        texts = [document_text(document) for document in cleaned]
        with self._state_lock:
            if generation != self._generation:
                raise _Cancelled()
            self.state = "building"
            self.done = 0
            self.total = len(texts)

        bar = None
        if show_progress:
            try:
                from tqdm import tqdm

                bar = tqdm(total=len(texts), unit="doc", desc="Embeddings")
            except ImportError:
                bar = None

        chunks = []
        try:
            for start in range(0, len(texts), ENCODE_CHUNK):
                with self._state_lock:
                    if generation != self._generation:
                        raise _Cancelled()
                batch = texts[start:start + ENCODE_CHUNK]
                with self._encode_lock:
                    vectors = _normalize(self.encoder.encode(batch))
                if vectors.shape[0] != len(batch):
                    raise ValueError("El codificador devolvió un número de vectores incorrecto")
                chunks.append(vectors)
                with self._state_lock:
                    if generation == self._generation:
                        self.done = start + len(batch)
                if bar is not None:
                    bar.update(len(batch))
        finally:
            if bar is not None:
                bar.close()
        return np.ascontiguousarray(np.vstack(chunks), dtype=np.float32)

    # ------------------------------------------------------------------ caché
    def _read_cache(self, path: Path, fingerprint: str, ids: list[str]) -> np.ndarray | None:
        meta_file = path / "meta.json"
        if not meta_file.is_file():
            return None
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if (
                meta.get("version") != CACHE_VERSION
                or meta.get("fingerprint") != fingerprint
                or meta.get("model") != self.model_name
            ):
                return None
            if json.loads((path / "ids.json").read_text(encoding="utf-8")) != ids:
                return None
            embeddings = np.load(path / "embeddings.npy", allow_pickle=False)
        except (OSError, ValueError, EOFError):
            return None
        if embeddings.dtype != np.float32 or embeddings.ndim != 2 or embeddings.shape[0] != len(ids):
            return None
        return np.ascontiguousarray(embeddings)

    def _write_cache(self, path: Path, fingerprint: str, ids: list[str], embeddings: np.ndarray) -> None:
        path.mkdir(parents=True, exist_ok=True)
        meta_file = path / "meta.json"
        # meta.json se escribe al final: sin él, una caché a medias no se lee.
        meta_file.unlink(missing_ok=True)
        np.save(path / "embeddings.npy", embeddings, allow_pickle=False)
        (path / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
        meta_file.write_text(json.dumps({
            "version": CACHE_VERSION,
            "model": self.model_name,
            "fingerprint": fingerprint,
            "count": len(ids),
            "dim": int(embeddings.shape[1]),
            "normalized": True,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2), encoding="utf-8")

    # --------------------------------------------------------------- búsqueda
    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Devuelve ``[(doc_id, coseno)]`` ordenado, listo para ``format_hits``.

        Lista vacía si la consulta está vacía o el índice no está listo.
        """
        if not str(query).strip() or k <= 0:
            return []
        with self._state_lock:
            index, ids = self._index, self._ids
        if index is None or not ids:
            return []
        with self._encode_lock:
            query_vector = _normalize(self.encoder.encode([str(query)]))
        if query_vector.shape[1] != index.d:
            raise ValueError("La dimensión de la consulta no coincide con la del índice")
        scores, positions = index.search(query_vector, min(k, index.ntotal))
        results = [
            (ids[position], float(score))
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        ]
        results.sort(key=lambda item: (-item[1], item[0]))
        return results


# ------------------------------------------------------------------- script
def load_corpus(dataset_name: str = "BeIR/fiqa", split: str = "corpus", limit: int | None = None) -> list[dict]:
    """Carga el corpus como ``SearchEngine.import_huggingface`` (mismo orden)."""
    from datasets import load_dataset

    corpus = load_dataset(dataset_name, "corpus", split=split)
    if limit is not None:
        corpus = corpus.select(range(min(limit, len(corpus))))
    return [dict(document) for document in corpus]


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("debe ser un entero positivo")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Genera la caché de embeddings del modelo dense para el servidor."
    )
    parser.add_argument("--build", action="store_true", help="carga el corpus y genera la caché")
    parser.add_argument("--limit", type=_positive_int, help="usa solo los primeros N documentos (pruebas)")
    parser.add_argument("--dataset", default="BeIR/fiqa")
    parser.add_argument("--split", default="corpus")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force", action="store_true", help="regenera aunque exista una caché válida")
    args = parser.parse_args(argv)

    if not args.build:
        parser.print_help()
        return 2

    started = time.perf_counter()
    print(f"Cargando {args.dataset}/{args.split}...", flush=True)
    documents = load_corpus(args.dataset, args.split, args.limit)
    print(f"{len(documents)} documentos.", flush=True)

    retriever = DenseRetriever(model_name=args.model, cache_dir=args.cache_dir, batch_size=args.batch_size)
    status = retriever.build(documents, force=args.force, show_progress=True)

    if status["origin"] == "cache":
        print("Ya existía una caché válida para este corpus; no se regeneró (usa --force).")
    print(f"Caché: {retriever.cache_location}")
    print(f"Tiempo total: {time.perf_counter() - started:.1f} s")
    if args.limit is not None:
        print(
            f"Nota: esta caché es solo de los primeros {len(documents)} documentos; "
            "el servidor con el corpus completo no la usará."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
