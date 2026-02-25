import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

public_router = APIRouter()
router = APIRouter()


@public_router.get("/health", tags=["diagnostic"])
def health() -> JSONResponse:
    return JSONResponse(status_code=200, content={"message": "OK"})
