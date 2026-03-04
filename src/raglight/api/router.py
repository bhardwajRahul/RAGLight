import os
import shutil
import tempfile
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..document_processing.document_processor_factory import DocumentProcessorFactory
from ..models.data_source_model import GitHubSource
from ..scrapper.github_scrapper import GithubScrapper


class GenerateRequest(BaseModel):
    question: str


class GenerateResponse(BaseModel):
    answer: str


class IngestRequest(BaseModel):
    data_path: Optional[str] = None
    file_paths: Optional[List[str]] = None
    github_url: Optional[str] = None
    github_branch: str = "main"


class IngestResponse(BaseModel):
    message: str


class CollectionsResponse(BaseModel):
    collections: list


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health():
        return {"status": "ok"}

    @router.post("/generate", response_model=GenerateResponse)
    async def generate(request: Request, body: GenerateRequest):
        pipeline = request.app.state.pipeline
        try:
            answer = await run_in_threadpool(pipeline.generate, body.question)
            return GenerateResponse(answer=answer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/ingest", response_model=IngestResponse)
    async def ingest(request: Request, body: IngestRequest):
        if not body.data_path and not body.file_paths and not body.github_url:
            raise HTTPException(
                status_code=400,
                detail="Provide at least one of: data_path, file_paths, github_url",
            )

        pipeline = request.app.state.pipeline
        vector_store = pipeline.get_vector_store()

        def _ingest_files(file_paths: List[str]):
            factory = DocumentProcessorFactory()
            missing = [fp for fp in file_paths if not os.path.isfile(fp)]
            if missing:
                raise ValueError(f"Files not found: {missing}")
            for fp in file_paths:
                chunks, classes = vector_store._process_file(
                    fp, factory, vector_store._flatten_metadata
                )
                if chunks:
                    vector_store.add_documents(chunks)
                if classes:
                    vector_store.add_class_documents(classes)

        def _do_ingest():
            if body.data_path:
                vector_store.ingest(data_path=body.data_path)
            if body.file_paths:
                _ingest_files(body.file_paths)
            if body.github_url:
                source = GitHubSource(url=body.github_url, branch=body.github_branch)
                scrapper = GithubScrapper()
                scrapper.set_repositories([source])
                repos_path = scrapper.clone_all()
                try:
                    vector_store.ingest(data_path=repos_path)
                finally:
                    shutil.rmtree(repos_path, ignore_errors=True)

        try:
            await run_in_threadpool(_do_ingest)
            return IngestResponse(message="Ingestion completed successfully")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/ingest/upload",
        response_model=IngestResponse,
        openapi_extra={
            "requestBody": {
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["files"],
                            "properties": {
                                "files": {
                                    "type": "array",
                                    "items": {"type": "string", "format": "binary"},
                                }
                            },
                        }
                    }
                }
            }
        },
    )
    async def ingest_upload(request: Request, files: List[UploadFile] = File(...)):
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        pipeline = request.app.state.pipeline
        vector_store = pipeline.get_vector_store()

        # Read all file contents while still in async context
        saved: List[tuple[str, bytes]] = []
        for upload in files:
            content = await upload.read()
            saved.append((upload.filename or "upload", content))

        def _do_upload():
            factory = DocumentProcessorFactory()
            tmp_dir = tempfile.mkdtemp(prefix="raglight_upload_")
            try:
                for filename, content in saved:
                    dest = os.path.join(tmp_dir, filename)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "wb") as f:
                        f.write(content)
                    chunks, classes = vector_store._process_file(
                        dest, factory, vector_store._flatten_metadata
                    )
                    if chunks:
                        vector_store.add_documents(chunks)
                    if classes:
                        vector_store.add_class_documents(classes)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        try:
            await run_in_threadpool(_do_upload)
            names = ", ".join(name for name, _ in saved)
            return IngestResponse(message=f"Ingested {len(saved)} file(s): {names}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/collections", response_model=CollectionsResponse)
    async def collections(request: Request):
        pipeline = request.app.state.pipeline
        vector_store = pipeline.get_vector_store()
        cols = await run_in_threadpool(vector_store.get_available_collections)
        return CollectionsResponse(collections=cols)

    return router
