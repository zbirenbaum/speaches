from pathlib import Path
from tempfile import NamedTemporaryFile

import gradio as gr

from speaches.config import Config
from speaches.ui.utils import http_client_from_gradio_req

DEFAULT_TEXT = "A rainbow is an optical phenomenon caused by refraction, internal reflection and dispersion of light in water droplets resulting in a continuous spectrum of light appearing in the sky."
SUPPORTED_FORMATS = ("pcm", "mp3", "wav", "flac", "opus", "aac")


def create_tts_tab(config: Config, api_key_input: gr.Textbox) -> None:
    async def update_model_dropdown(api_key: str, request: gr.Request) -> gr.Dropdown:
        http_client = http_client_from_gradio_req(request, config, api_key or None)
        res = await http_client.get("/v1/models", params={"task": "tts"})
        res.raise_for_status()
        models = res.json().get("data", [])
        model_ids = [m["id"] for m in models]
        return gr.Dropdown(choices=model_ids, label="Model")

    async def handle_audio_speech(
        text: str,
        model: str,
        response_format: str,
        api_key: str,
        request: gr.Request,
    ) -> Path:
        http_client = http_client_from_gradio_req(request, config, api_key or None)
        res = await http_client.post(
            "/v1/audio/speech",
            json={
                "model": model,
                "input": text,
                "response_format": response_format,
            },
        )
        res.raise_for_status()
        audio_bytes = res.content
        with NamedTemporaryFile(suffix=f".{response_format}", delete=False) as file:
            file.write(audio_bytes)
            file_path = Path(file.name)
        return file_path

    with gr.Tab(label="Text-to-Speech") as tab:
        text = gr.Textbox(label="Input Text", value=DEFAULT_TEXT, lines=3)
        model_dropdown = gr.Dropdown(choices=[], label="Model")
        response_format_dropdown = gr.Dropdown(
            choices=SUPPORTED_FORMATS,
            label="Response Format",
            value="wav",
        )
        button = gr.Button("Generate Speech")
        output = gr.Audio(type="filepath")
        button.click(
            handle_audio_speech,
            [text, model_dropdown, response_format_dropdown, api_key_input],
            output,
        )

        tab.select(update_model_dropdown, inputs=[api_key_input], outputs=model_dropdown)
