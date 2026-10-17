from functools import lru_cache

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.retrievers.bm25 import BM25Retriever

import config
from index import get_index, load_docstore
from metadata import get_llm


@lru_cache
def reranker():
    return SentenceTransformerRerank(model=config.RERANK_MODEL, top_n=config.RERANK_TOP_N, device=config.DEVICE)


def _matches(meta: dict, paper_id, section_type, year_min) -> bool:
    if paper_id and meta.get("paper_id") != paper_id:
        return False
    if section_type and meta.get("section_type") != section_type:
        return False
    if year_min and (meta.get("year") or 0) < year_min:
        return False
    return True


def retrieve(query: str, paper_id: str | None = None, section_type: str | None = None, year_min: int | None = None):
    nodes = [n for n in load_docstore().docs.values() if _matches(n.metadata, paper_id, section_type, year_min)]
    if not nodes:
        return []

    fl = []
    if paper_id:
        fl.append(MetadataFilter(key="paper_id", value=paper_id))
    if section_type:
        fl.append(MetadataFilter(key="section_type", value=section_type))
    if year_min:
        fl.append(MetadataFilter(key="year", value=year_min, operator=FilterOperator.GTE))
    filters = MetadataFilters(filters=fl) if fl else None

    dense = get_index().as_retriever(similarity_top_k=config.TOP_K_DENSE, filters=filters)
    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=min(config.TOP_K_BM25, len(nodes)))
    fusion = QueryFusionRetriever([dense, bm25], 
                                  similarity_top_k=config.FUSION_TOP_K, 
                                  num_queries=1, 
                                  mode="reciprocal_rerank", 
                                  use_async=False, 
                                  llm=get_llm())
    return reranker().postprocess_nodes(fusion.retrieve(query), query_str=query)
