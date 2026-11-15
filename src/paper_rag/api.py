import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config
from .generate import answer_events
from .index import client, embed_model, list_papers
from .metadata import find_paper, read_json
from .retrieve import reranker
from .sections import SECTION_TYPES


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the models and open the store once, before the first request
    client()
    embed_model()
    reranker()
    yield


app = FastAPI(title="paper-rag", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    paper: str | None = None
    section_type: str | None = None
    year_min: int | None = None


@app.get("/papers")
def papers():
    items = sorted(list_papers(), key=lambda m: (m.get("year") or 0, m["paper_id"]))
    return {"papers": items, "section_types": list(SECTION_TYPES)}


def _sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/ask")
def ask(req: AskRequest):
    if req.section_type and req.section_type not in SECTION_TYPES:
        raise HTTPException(422, f"Unknown section_type: {req.section_type}")
    try:
        paper_id = find_paper(req.paper) if req.paper else None
    except ValueError as e:
        raise HTTPException(400, str(e))

    def stream():
        try:
            for event in answer_events(req.question, paper_id, req.section_type, req.year_min):
                yield _sse(event)
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/pdf/{paper_id:path}")
def pdf(paper_id: str):
    entry = read_json(config.REGISTRY_PATH).get(paper_id)
    if not entry or not entry.get("source"):
        raise HTTPException(404, "Unknown paper")
    root = config.PDF_DIR.resolve()
    path = (root / entry["source"]).resolve()
    if path.parent != root or not path.is_file():
        raise HTTPException(404, "PDF not found")
    return FileResponse(path, media_type="application/pdf", filename=path.name,
                        content_disposition_type="inline")
