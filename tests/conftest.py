from collections.abc import AsyncGenerator, Generator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
import logging
from typing import Literal, Protocol

from fastapi.testclient import TestClient
import httpx
from httpx import ASGITransport, AsyncClient
from openai import AsyncOpenAI
import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from speaches.config import Config
from speaches.dependencies import get_config
from speaches.main import create_app

DISABLE_LOGGERS = ["multipart.multipart"]
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_VAD_TTL = 0
DEFAULT_CONFIG = Config(
    vad_model_ttl=DEFAULT_VAD_TTL,
    enable_ui=False,
    loopback_host_url=None,
)
TIMEOUT = httpx.Timeout(15.0)


def pytest_configure() -> None:
    for logger_name in DISABLE_LOGGERS:
        logger = logging.getLogger(logger_name)
        logger.disabled = True


@pytest.fixture
def client() -> Generator[TestClient]:
    with TestClient(create_app()) as client:
        yield client


class AclientFactory(Protocol):
    def __call__(self, config: Config = DEFAULT_CONFIG) -> AbstractAsyncContextManager[AsyncClient]: ...


@pytest_asyncio.fixture()
async def aclient_factory(mocker: MockerFixture) -> AclientFactory:
    @asynccontextmanager
    async def inner(config: Config = DEFAULT_CONFIG) -> AsyncGenerator[AsyncClient]:
        mocker.patch("speaches.dependencies.get_config", return_value=config)
        mocker.patch("speaches.main.get_config", return_value=config)

        app = create_app()
        app.dependency_overrides[get_config] = lambda: config
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=TIMEOUT) as aclient:
            yield aclient

    return inner


@pytest_asyncio.fixture()
async def aclient(aclient_factory: AclientFactory) -> AsyncGenerator[AsyncClient]:
    async with aclient_factory() as aclient:
        yield aclient


@pytest.fixture
def openai_client(aclient: AsyncClient) -> AsyncOpenAI:
    return AsyncOpenAI(api_key="cant-be-empty", http_client=aclient, max_retries=0)


@pytest.fixture
def actual_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=OPENAI_BASE_URL,
        max_retries=0,
    )


@pytest_asyncio.fixture()
async def dynamic_openai_client(
    target: Literal["speaches", "openai"], aclient_factory: AclientFactory
) -> AsyncGenerator[AsyncOpenAI]:
    assert target in ["speaches", "openai"]
    if target == "openai":
        yield AsyncOpenAI(base_url=OPENAI_BASE_URL, max_retries=0)
    elif target == "speaches":
        async with aclient_factory() as aclient:
            yield AsyncOpenAI(api_key="cant-be-empty", http_client=aclient, max_retries=0)
