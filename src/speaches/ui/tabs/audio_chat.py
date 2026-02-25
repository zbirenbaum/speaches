import base64
from collections.abc import AsyncGenerator
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, TypedDict

import gradio as gr
from httpx_sse import aconnect_sse
from pydantic import BaseModel

from speaches.config import Config
from speaches.ui.utils import http_client_from_gradio_req

logger = logging.getLogger(__name__)

type Modality = Literal["text", "audio"]


class GradioMessage(TypedDict):
    text: str
    files: list[str]


class VoiceChatState(BaseModel):
    messages: list[dict] = []


def gradio_message_to_messages(gradio_message: GradioMessage) -> dict:
    content: list[dict] = []

    if len(gradio_message["text"]) > 0:
        content.append({"type": "text", "text": gradio_message["text"]})

    for file_path in gradio_message["files"]:
        if not file_path:
            continue
        suffix = Path(file_path).suffix.lower()
        file_bytes = Path(file_path).read_bytes()
        b64 = base64.b64encode(file_bytes).decode("utf-8")

        if suffix in (".wav", ".mp3", ".flac", ".ogg"):
            data_url = f"data:audio/{suffix.lstrip('.')};base64,{b64}"
            content.append({"type": "audio_url", "audio_url": {"url": data_url}})
        elif suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            data_url = f"data:image/{suffix.lstrip('.')};base64,{b64}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        elif suffix in (".mp4", ".webm", ".avi"):
            data_url = f"data:video/{suffix.lstrip('.')};base64,{b64}"
            content.append({"type": "video_url", "video_url": {"url": data_url}})

    return {"role": "user", "content": content}


def create_audio_chat_tab(config: Config, api_key_input: gr.Textbox) -> None:
    async def create_reply(
        message: GradioMessage,
        _history: list[gr.ChatMessage],
        model: str,
        modality: str,
        stream: bool,
        state: VoiceChatState,
        api_key: str,
        request: gr.Request,
    ) -> AsyncGenerator[list[gr.ChatMessage] | gr.ChatMessage]:
        http_client = http_client_from_gradio_req(request, config, api_key or None)

        user_msg = gradio_message_to_messages(message)
        state.messages.append(user_msg)

        modalities = [modality]

        body = {
            "model": model,
            "messages": state.messages,
            "modalities": modalities,
            "stream": stream,
            "max_tokens": 2048,
        }

        if stream:
            text_content = ""
            async with aconnect_sse(http_client, "POST", "/v1/chat/completions", json=body) as event_source:
                async for sse in event_source.aiter_sse():
                    data = sse.data
                    if data in {"[DONE]", "data: [DONE]"}:
                        break
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})

                    # Text modality
                    if delta.get("content"):
                        text_content += delta["content"]
                        yield gr.ChatMessage(role="assistant", content=text_content)

            if text_content:
                state.messages.append({"role": "assistant", "content": text_content})
        else:
            res = await http_client.post("/v1/chat/completions", json=body)
            res.raise_for_status()
            result = res.json()
            choice = result["choices"][0]
            msg = choice["message"]

            if msg.get("audio") and msg["audio"].get("data"):
                audio_bytes = base64.b64decode(msg["audio"]["data"])
                with NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = Path(tmp_file.name)

                transcript = msg.get("audio", {}).get("transcript", "")
                messages_out = []
                if transcript:
                    messages_out.append(gr.ChatMessage(role="assistant", content=transcript))
                messages_out.append(gr.ChatMessage(role="assistant", content=gr.FileData(path=str(tmp_path))))
                state.messages.append({"role": "assistant", "content": transcript or "[audio]"})
                yield messages_out
            elif msg.get("content"):
                text = msg["content"]
                state.messages.append({"role": "assistant", "content": text})
                yield gr.ChatMessage(role="assistant", content=text)

    async def update_chat_model_dropdown(api_key: str, request: gr.Request) -> gr.Dropdown:
        http_client = http_client_from_gradio_req(request, config, api_key or None)
        res = await http_client.get("/v1/models", params={"task": "chat"})
        res.raise_for_status()
        models = res.json().get("data", [])
        model_ids = [m["id"] for m in models]
        return gr.Dropdown(
            choices=model_ids,
            label="Chat Model",
            value=model_ids[0] if model_ids else None,
        )

    with gr.Tab(label="Audio Chat") as tab:
        state = gr.State(VoiceChatState())
        chat_model_dropdown = gr.Dropdown(
            choices=[],
            label="Chat Model",
            value=None,
        )
        modality_dropdown = gr.Dropdown(
            choices=["text", "audio"],
            label="Output Modality",
            value="text",
        )
        stream_checkbox = gr.Checkbox(label="Stream", value=True)
        gr.ChatInterface(
            type="messages",
            multimodal=True,
            fn=create_reply,
            textbox=gr.MultimodalTextbox(
                interactive=True,
                file_count="multiple",
                placeholder="Enter message or upload file (audio, image, video)...",
                sources=["microphone", "upload"],
            ),
            additional_inputs=[chat_model_dropdown, modality_dropdown, stream_checkbox, state, api_key_input],
        )

        tab.select(
            update_chat_model_dropdown,
            inputs=[api_key_input],
            outputs=[chat_model_dropdown],
        )
