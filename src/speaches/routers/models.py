import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from speaches.dependencies import BackendRegistryDependency

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])


@router.get("/v1/models")
def list_models(registry: BackendRegistryDependency, task: str | None = None) -> JSONResponse:
    backends = registry.list_models()
    if task:
        backends = [b for b in backends if task in b.capabilities]
    data = [
        {
            "id": b.id,
            "object": "model",
            "created": 0,
            "owned_by": "backend",
            "capabilities": b.capabilities,
            "input_modalities": b.input_modalities,
            "output_modalities": b.output_modalities,
        }
        for b in backends
    ]
    return JSONResponse(content={"data": data, "object": "list"})


@router.get("/v1/models/{model_id:path}")
def get_model(registry: BackendRegistryDependency, model_id: str) -> JSONResponse:
    backend = registry.get_backend(model_id)
    if backend is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return JSONResponse(
        content={
            "id": backend.id,
            "object": "model",
            "created": 0,
            "owned_by": "backend",
            "capabilities": backend.capabilities,
            "input_modalities": backend.input_modalities,
            "output_modalities": backend.output_modalities,
        }
    )
