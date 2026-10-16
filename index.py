from functools import lru_cache

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from metadata import PaperMeta


@lru_cache
def embed_model():
    return HuggingFaceEmbedding(model_name=config.EMBED_MODEL, device=config.DEVICE, embed_batch_size=16)


def _vector_store() -> tuple[ChromaVectorStore, chromadb.Collection]:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    coll = client.get_or_create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})
    return ChromaVectorStore(chroma_collection=coll), coll


def get_index() -> VectorStoreIndex:
    vs, _ = _vector_store()
    return VectorStoreIndex.from_vector_store(vs, embed_model=embed_model())


def add_paper(meta: PaperMeta, nodes) -> None:
    bad = [n for n in nodes if n.metadata.get("paper_id") != meta.paper_id]
    if bad:
        raise ValueError(f"{len(bad)} nodes have mismatched paper_id: {bad[0].metadata.get('paper_id')} != {meta.paper_id}")

    vs, coll = _vector_store()

    coll.delete(where={"paper_id": meta.paper_id})
    coll.delete(where={"content_hash": meta.content_hash})

    index = VectorStoreIndex.from_vector_store(vs, embed_model=embed_model(), show_progress=True)
    index.insert_nodes(nodes)
