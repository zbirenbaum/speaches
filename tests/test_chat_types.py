from speaches.types.chat import (
    AudioURL,
    ChatCompletionContentPartAudioUrlParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartVideoUrlParam,
    ChatCompletionUserMessageParam,
    CompletionCreateParamsBase,
    ImageURL,
    InputAudio,
    VideoURL,
)


class TestNewContentPartTypes:
    def test_audio_url_model(self) -> None:
        audio_url = AudioURL(url="data:audio/wav;base64,AAAA")
        assert audio_url.url == "data:audio/wav;base64,AAAA"

    def test_video_url_model(self) -> None:
        video_url = VideoURL(url="data:video/mp4;base64,BBBB")
        assert video_url.url == "data:video/mp4;base64,BBBB"

    def test_audio_url_content_part(self) -> None:
        part = ChatCompletionContentPartAudioUrlParam(
            audio_url=AudioURL(url="data:audio/wav;base64,test"),
            type="audio_url",
        )
        assert part.type == "audio_url"
        assert part.audio_url.url == "data:audio/wav;base64,test"

    def test_video_url_content_part(self) -> None:
        part = ChatCompletionContentPartVideoUrlParam(
            video_url=VideoURL(url="data:video/mp4;base64,test"),
            type="video_url",
        )
        assert part.type == "video_url"
        assert part.video_url.url == "data:video/mp4;base64,test"


class TestUserMessageWithNewTypes:
    def test_user_message_with_audio_url(self) -> None:
        msg = ChatCompletionUserMessageParam(
            role="user",
            content=[
                ChatCompletionContentPartTextParam(text="What is this?", type="text"),
                ChatCompletionContentPartAudioUrlParam(
                    audio_url=AudioURL(url="data:audio/wav;base64,AAAA"),
                    type="audio_url",
                ),
            ],
        )
        assert len(msg.content) == 2

    def test_user_message_with_video_url(self) -> None:
        msg = ChatCompletionUserMessageParam(
            role="user",
            content=[
                ChatCompletionContentPartTextParam(text="Describe this video", type="text"),
                ChatCompletionContentPartVideoUrlParam(
                    video_url=VideoURL(url="data:video/mp4;base64,BBBB"),
                    type="video_url",
                ),
            ],
        )
        assert len(msg.content) == 2

    def test_user_message_with_all_modalities(self) -> None:
        msg = ChatCompletionUserMessageParam(
            role="user",
            content=[
                ChatCompletionContentPartTextParam(text="Analyze all", type="text"),
                ChatCompletionContentPartImageParam(
                    image_url=ImageURL(url="data:image/png;base64,IMG"),
                    type="image_url",
                ),
                ChatCompletionContentPartInputAudioParam(
                    input_audio=InputAudio(data="base64data", format="wav"),
                    type="input_audio",
                ),
                ChatCompletionContentPartAudioUrlParam(
                    audio_url=AudioURL(url="data:audio/wav;base64,AUD"),
                    type="audio_url",
                ),
                ChatCompletionContentPartVideoUrlParam(
                    video_url=VideoURL(url="data:video/mp4;base64,VID"),
                    type="video_url",
                ),
            ],
        )
        assert len(msg.content) == 5

    def test_user_message_with_string_content(self) -> None:
        msg = ChatCompletionUserMessageParam(role="user", content="Hello")
        assert msg.content == "Hello"


class TestCompletionCreateParams:
    def test_modalities_field(self) -> None:
        params = CompletionCreateParamsBase(
            model="test-model",
            messages=[],
            modalities=["text", "audio"],
        )
        assert params.modalities == ["text", "audio"]

    def test_modalities_text_only(self) -> None:
        params = CompletionCreateParamsBase(
            model="test-model",
            messages=[],
            modalities=["text"],
        )
        assert params.modalities == ["text"]

    def test_modalities_audio_only(self) -> None:
        params = CompletionCreateParamsBase(
            model="test-model",
            messages=[],
            modalities=["audio"],
        )
        assert params.modalities == ["audio"]

    def test_modalities_default_none(self) -> None:
        params = CompletionCreateParamsBase(
            model="test-model",
            messages=[],
        )
        assert params.modalities is None

    def test_messages_with_multimodal_content(self) -> None:
        params = CompletionCreateParamsBase(
            model="test-model",
            messages=[
                ChatCompletionUserMessageParam(
                    role="user",
                    content=[
                        ChatCompletionContentPartTextParam(text="Hello", type="text"),
                        ChatCompletionContentPartAudioUrlParam(
                            audio_url=AudioURL(url="data:audio/wav;base64,X"),
                            type="audio_url",
                        ),
                    ],
                ),
            ],
            modalities=["text"],
        )
        assert len(params.messages) == 1
