from functools import lru_cache

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from metadata import PaperMeta, load_registry


@lru_cache
def embed_model():
    return HuggingFaceEmbedding(model_name=config.EMBED_MODEL, device=config.DEVICE, embed_batch_size=16)


def load_docstore() -> SimpleDocumentStore:
    if config.DOCSTORE_PATH.exists():
        return SimpleDocumentStore.from_persist_path(str(config.DOCSTORE_PATH))
    return SimpleDocumentStore()


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
    docstore = load_docstore()

    coll.delete(where={"paper_id": meta.paper_id})
    coll.delete(where={"content_hash": meta.content_hash})
    
    for node_id, node in list(docstore.docs.items()):
        if node.metadata.get("paper_id") == meta.paper_id or node.metadata.get("content_hash") == meta.content_hash:
            docstore.delete_document(node_id)

    index = VectorStoreIndex.from_vector_store(vs, embed_model=embed_model(), show_progress=True)
    index.insert_nodes(nodes)

    docstore.add_documents(nodes)
    config.DOCSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    docstore.persist(str(config.DOCSTORE_PATH))


def list_papers() -> list[dict]:
    counts: dict[str, int] = {}
    for node in load_docstore().docs.values():
        pid = node.metadata.get("paper_id", "?")
        counts[pid] = counts.get(pid, 0) + 1
    return [{**m, "chunks": counts.get(pid, 0)} for pid, m in load_registry().items()]
