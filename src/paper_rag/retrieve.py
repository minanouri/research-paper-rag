from functools import lru_cache

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, TextNode
from qdrant_client import models

from . import config
from .index import DENSE, SPARSE, bm25_query, client, embed_model, has_index

# Reciprocal rank fusion constant
RRF_K = 60


@lru_cache
def reranker():
    rr = SentenceTransformerRerank(model=config.RERANK_MODEL, top_n=config.RERANK_TOP_N, device=config.DEVICE)
    if config.USE_FP16:
        # SentenceTransformerRerank has no dtype option, so cast its CrossEncoder directly
        rr._model.half()
    return rr


def _filter(paper_id, section_type, year_min) -> models.Filter | None:
    must = []
    if paper_id:
        must.append(models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id)))
    if section_type:
        must.append(models.FieldCondition(key="section_type", match=models.MatchValue(value=section_type)))
    if year_min:
        must.append(models.FieldCondition(key="year", range=models.Range(gte=year_min)))
    return models.Filter(must=must) if must else None


def _prefetches(query: str, flt) -> list[models.Prefetch]:
    """The dense and BM25 candidate searches (BM25 is skipped if the query has no indexable terms)."""
    out = [models.Prefetch(query=embed_model().get_query_embedding(query), using=DENSE,
                           limit=config.TOP_K_DENSE, filter=flt)]
    sparse = bm25_query(query)
    if sparse.indices:
        out.append(models.Prefetch(query=sparse, using=SPARSE, limit=config.TOP_K_BM25, filter=flt))
    return out


def _nodes(points) -> list[NodeWithScore]:
    return [NodeWithScore(node=TextNode.from_json(p.payload["_node"]), score=p.score) for p in points]


def _fuse(prefetches, flt) -> list[NodeWithScore]:
    res = client().query_points(config.COLLECTION, prefetch=prefetches, query=models.RrfQuery(rrf=models.Rrf(k=RRF_K)),
                                query_filter=flt, limit=config.FUSION_TOP_K, with_payload=["_node"])
    return _nodes(res.points)


def hybrid_search(query: str, paper_id: str | None = None, section_type: str | None = None,
                  year_min: int | None = None) -> list[NodeWithScore]:
    if not has_index():
        return []
    flt = _filter(paper_id, section_type, year_min)
    return _fuse(_prefetches(query, flt), flt)


def search_stages(query: str, paper_id: str | None = None, section_type: str | None = None,
                  year_min: int | None = None) -> dict[str, list[NodeWithScore]]:
    """Dense, BM25 and fused results for one query, so the eval can score each stage."""
    if not has_index():
        return {}
    flt = _filter(paper_id, section_type, year_min)
    prefetches = _prefetches(query, flt)
    stages = {"dense": [], "bm25": []}
    for p in prefetches:
        res = client().query_points(config.COLLECTION, query=p.query, using=p.using, limit=p.limit,
                                    query_filter=flt, with_payload=["_node"])
        stages["dense" if p.using == DENSE else "bm25"] = _nodes(res.points)
    stages["fusion"] = _fuse(prefetches, flt)
    return stages


def retrieve(query: str, paper_id: str | None = None, section_type: str | None = None, year_min: int | None = None):
    nodes = hybrid_search(query, paper_id, section_type, year_min)
    return reranker().postprocess_nodes(nodes, query_str=query) if nodes else []
