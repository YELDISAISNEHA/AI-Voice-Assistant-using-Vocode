import asyncio
import signal
import io
import google.generativeai as genai
from google.cloud import texttospeech

from pydantic_settings import BaseSettings, SettingsConfigDict
from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.logging import configure_pretty_logging
from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer
from vocode.streaming.models.synthesizer import SynthesizerConfig, AudioEncoding
from vocode.streaming.streaming_conversation import StreamingConversation

configure_pretty_logging()

class Settings(BaseSettings):
    gemini_api_key: str = "API_KEY"
    deepgram_api_key: str = "API_KEY"
    google_credentials_json: str = r"C:/Users/DELL/OneDrive/Documents/B191333/Blackcoffer/Training_tasks/18.Vocode/vocodechatbot-3c549ed703c9.json"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()

class GeminiAgent(BaseAgent):
    def __init__(self, api_key: str):
        super().__init__(AgentConfig(type="custom"))
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")

    async def respond(self, message: BaseMessage, conversation_id=None, **kwargs):
        response = await asyncio.to_thread(self.model.generate_content, message.text)
        text = getattr(response, "text", None) or "Sorry, I didn’t catch that."
        return BaseMessage(text=text)

class GoogleTTSSynthesizer(BaseSynthesizer):
    supports_streaming = True

    def __init__(self, config: SynthesizerConfig):
        super().__init__(config)
        self.client = texttospeech.TextToSpeechClient.from_service_account_file(
            settings.google_credentials_json
        )

    async def create_speech(self, message: BaseMessage, conversation_id=None, **kwargs):
        synthesis_input = texttospeech.SynthesisInput(text=message.text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )

        response = await asyncio.to_thread(
            self.client.synthesize_speech,
            synthesis_input,
            voice,
            audio_config,
        )

        audio_stream = io.BytesIO(response.audio_content)
        chunk_size = 4096
        while True:
            chunk = audio_stream.read(chunk_size)
            if not chunk:
                break
            yield chunk
            await asyncio.sleep(0)

    async def synthesize_speech(self, message: BaseMessage, conversation_id=None, **kwargs):
        synthesis_input = texttospeech.SynthesisInput(text=message.text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )

        response = await asyncio.to_thread(
            self.client.synthesize_speech,
            synthesis_input,
            voice,
            audio_config,
        )
        return response.audio_content

async def main():
    mic, speaker = create_streaming_microphone_input_and_speaker_output(
        use_default_devices=True
    )

    transcriber = DeepgramTranscriber(
        DeepgramTranscriberConfig.from_input_device(
            mic,
            endpointing_config=PunctuationEndpointingConfig(),
            api_key=settings.deepgram_api_key,
        )
    )

    agent = GeminiAgent(api_key=settings.gemini_api_key)
    synthesizer = GoogleTTSSynthesizer(
        SynthesizerConfig(
            sampling_rate=speaker.sampling_rate,
            audio_encoding=AudioEncoding.LINEAR16,
        )
    )

    conversation = StreamingConversation(
        output_device=speaker,
        transcriber=transcriber,
        agent=agent,
        synthesizer=synthesizer,
    )

    await conversation.start()
    print("Conversation started — speak into mic. Ctrl+C to stop.")

    signal.signal(
        signal.SIGINT,
        lambda _0, _1: asyncio.create_task(conversation.terminate()),
    )

    try:
        while conversation.is_active():
            chunk = await mic.get_audio()
            conversation.receive_audio(chunk)
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main())

