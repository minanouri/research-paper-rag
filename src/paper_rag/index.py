import atexit
import warnings
from collections import Counter
from functools import lru_cache
from typing import Iterator

import bm25s
import mmh3
import Stemmer
import torch
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient, models

from . import config
from .metadata import PaperMeta, read_json

# Named vectors in the Qdrant collection
DENSE, SPARSE = "dense", "bm25"
# BM25 parameters
K1, B = 1.5, 0.75
# Payload fields used as retrieval filters
FILTER_FIELDS = {"paper_id": models.PayloadSchemaType.KEYWORD,
                 "section_type": models.PayloadSchemaType.KEYWORD,
                 "year": models.PayloadSchemaType.INTEGER}

_stemmer = Stemmer.Stemmer("english")


@lru_cache
def embed_model():
    model_kwargs = {"dtype": torch.float16} if config.USE_FP16 else {}
    return HuggingFaceEmbedding(model_name=config.EMBED_MODEL, device=config.DEVICE, embed_batch_size=16,
                                model_kwargs=model_kwargs)


@lru_cache
def client() -> QdrantClient:
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL)
    config.QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    local = QdrantClient(path=str(config.QDRANT_DIR))
    # Close before interpreter shutdown
    atexit.register(local.close)
    return local


def has_index() -> bool:
    return client().collection_exists(config.COLLECTION)


def _tokens(texts: list[str]) -> list[list[str]]:
    return bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer, return_ids=False, show_progress=False)


def _bm25_doc(tokens: list[str], avgdl: float) -> models.SparseVector:
    tf = Counter(mmh3.hash(t, signed=False) for t in tokens)
    norm = K1 * (1 - B + B * len(tokens) / avgdl)
    return models.SparseVector(indices=list(tf), values=[c / (c + norm) for c in tf.values()])


def bm25_query(query: str) -> models.SparseVector:
    tf = Counter(mmh3.hash(t, signed=False) for t in _tokens([query])[0])
    return models.SparseVector(indices=list(tf), values=[float(c) for c in tf.values()])


def _ensure_collection(dim: int) -> None:
    if has_index():
        return
    client().create_collection(
        config.COLLECTION,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Local mode warns that payload indexes have no effect there
        for field, schema in FILTER_FIELDS.items():
            client().create_payload_index(config.COLLECTION, field, schema)


def _point(node, vector: list[float]) -> models.PointStruct:
    return models.PointStruct(id=node.node_id, vector={DENSE: vector},
                              payload={**node.metadata, "_node": node.to_json()})


def iter_nodes() -> Iterator[TextNode]:
    if not has_index():
        return
    offset = None
    while True:
        points, offset = client().scroll(config.COLLECTION, limit=256, offset=offset, with_payload=["_node"])
        for p in points:
            yield TextNode.from_json(p.payload["_node"])
        if offset is None:
            return


def refresh_bm25() -> None:
    nodes = list(iter_nodes())
    if not nodes:
        return
    tokens = _tokens([n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes])
    avgdl = sum(map(len, tokens)) / len(tokens)
    points = [models.PointVectors(id=n.node_id, vector={SPARSE: _bm25_doc(t, avgdl)}) for n, t in zip(nodes, tokens)]
    for i in range(0, len(points), 256):
        client().update_vectors(config.COLLECTION, points=points[i:i + 256], wait=True)


def add_paper(meta: PaperMeta, nodes) -> None:
    bad = [n for n in nodes if n.metadata.get("paper_id") != meta.paper_id]
    if bad:
        raise ValueError(f"{len(bad)} nodes have mismatched paper_id: {bad[0].metadata.get('paper_id')} != {meta.paper_id}")

    vectors = embed_model().get_text_embedding_batch(
        [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes], show_progress=True)

    if has_index():
        old = models.Filter(should=[
            models.FieldCondition(key="paper_id", match=models.MatchValue(value=meta.paper_id)),
            models.FieldCondition(key="content_hash", match=models.MatchValue(value=meta.content_hash))])
        client().delete(config.COLLECTION, points_selector=models.FilterSelector(filter=old), wait=True)
    if nodes:
        _ensure_collection(len(vectors[0]))
        client().upload_points(config.COLLECTION, [_point(n, v) for n, v in zip(nodes, vectors)], wait=True)
    refresh_bm25()


def list_papers() -> list[dict]:
    def chunks(pid: str) -> int:
        if not has_index():
            return 0
        flt = models.Filter(must=[models.FieldCondition(key="paper_id", match=models.MatchValue(value=pid))])
        return client().count(config.COLLECTION, count_filter=flt, exact=True).count

    return [{**m, "chunks": chunks(pid)} for pid, m in read_json(config.REGISTRY_PATH).items()]
