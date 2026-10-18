from pathlib import Path
import torch


DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"
DOCSTORE_PATH = DATA_DIR / "docstore.json"
REGISTRY_PATH = DATA_DIR / "papers.json"
META_CACHE_PATH = DATA_DIR / "meta_cache.json"
COLLECTION = "papers"

# Models
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "qwen3:8b"
CONTEXT_WINDOW = 8192

# Metadata resolution
ALLOW_NETWORK = True
USE_LLM_META = True
CONTACT_EMAIL = ""

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SKIP_SECTIONS = ("references", "bibliography", 
                 "acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements", 
                 "declaration of competing interest", "declarations of competing interest", 
                 "conflict of interest", "conflicts of interest", 
                 "credit authorship contribution statement", "author contributions")

# Retrieval
TOP_K_DENSE = 20
TOP_K_BM25 = 20
FUSION_TOP_K = 20
RERANK_TOP_N = 5