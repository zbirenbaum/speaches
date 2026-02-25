from collections.abc import AsyncGenerator
import logging
from pathlib import Path

import gradio as gr
import httpx

from speaches.config import Config
from speaches.ui.utils import http_client_from_gradio_req
from speaches.utils import APIProxyError, format_api_proxy_error

logger = logging.getLogger(__name__)

TRANSCRIPTION_ENDPOINT = "/v1/audio/transcriptions"

type ResponseFormat = str
RESPONSE_FORMATS = ("text", "json", "verbose_json")
DEFAULT_RESPONSE_FORMAT: ResponseFormat = "text"


def create_stt_tab(config: Config, api_key_input: gr.Textbox) -> None:
    async def update_model_dropdown(api_key: str, request: gr.Request) -> gr.Dropdown:
        http_client = http_client_from_gradio_req(request, config, api_key or None)
        res = await http_client.get("/v1/models", params={"task": "stt"})
        res.raise_for_status()
        models = res.json().get("data", [])
        model_ids = [m["id"] for m in models]
        return gr.Dropdown(choices=model_ids, label="Model")

    async def audio_task(
        http_client: httpx.AsyncClient,
        file_path: str,
        response_format: ResponseFormat,
        temperature: float,
        model: str,
    ) -> str:
        try:
            if not file_path:
                msg = "No audio file provided. Please record or upload audio."
                raise APIProxyError(msg, suggestions=["Please record or upload an audio file."])
            with Path(file_path).open("rb") as file:  # noqa: ASYNC230
                response = await http_client.post(
                    TRANSCRIPTION_ENDPOINT,
                    files={"file": file},
                    data={
                        "model": model,
                        "response_format": response_format,
                        "temperature": temperature,
                    },
                )
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.exception("STT audio_task error")
            if not isinstance(e, APIProxyError):
                e = APIProxyError(str(e))
            return format_api_proxy_error(e, context="audio_task")

    async def handler(
        file_path: str,
        model: str,
        response_format: ResponseFormat,
        temperature: float,
        api_key: str,
        request: gr.Request,
    ) -> AsyncGenerator[str]:
        try:
            if not file_path:
                msg = "No audio file provided. Please record or upload audio."
                raise APIProxyError(msg, suggestions=["Please record or upload an audio file."])
            http_client = http_client_from_gradio_req(request, config, api_key or None)
            result = await audio_task(http_client, file_path, response_format, temperature, model)
            yield result
        except Exception as e:
            logger.exception("STT handler error")
            if not isinstance(e, APIProxyError):
                e = APIProxyError(str(e))
            yield format_api_proxy_error(e, context="handler")

    with gr.Tab(label="Speech-to-Text") as tab:
        audio = gr.Audio(type="filepath")
        model_dropdown = gr.Dropdown(choices=[], label="Model")
        response_format = gr.Dropdown(choices=RESPONSE_FORMATS, label="Response Format", value=DEFAULT_RESPONSE_FORMAT)
        temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Temperature", value=0.0)
        button = gr.Button("Generate")
        output = gr.Textbox()

        button.click(
            handler,
            [audio, model_dropdown, response_format, temperature_slider, api_key_input],
            output,
        )

        tab.select(update_model_dropdown, inputs=[api_key_input], outputs=model_dropdown)
