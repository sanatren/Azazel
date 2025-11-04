import os
import tempfile
import logging
from openai import OpenAI
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

class AudioHandler:
    def __init__(self, api_key):
        """Initialize AudioHandler with OpenAI API key"""
        if not api_key or not api_key.startswith('sk-'):
            raise ValueError("Invalid OpenAI API key format. Key should start with 'sk-'")

        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Transcribe audio using OpenAI Whisper API

        Args:
            audio_path: Path to the audio file

        Returns:
            Transcribed text or None if error
        """
        try:
            if not audio_path or not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return None

            # Use OpenAI API for transcription
            with open(audio_path, "rb") as audio_file:
                logger.info("Converting speech to text using OpenAI Whisper API...")
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

            logger.info("Audio transcription successful")
            return transcript.text

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return None

        finally:
            # Clean up temp file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logger.debug(f"Removed temporary file: {audio_path}")
                except Exception as e:
                    logger.warning(f"Could not remove temporary file: {e}")

    def transcribe_audio_bytes(self, audio_bytes: bytes, file_extension: str = "wav") -> Optional[str]:
        """Transcribe audio from bytes

        Args:
            audio_bytes: Audio file bytes
            file_extension: File extension (wav, mp3, m4a, etc.)

        Returns:
            Transcribed text or None if error
        """
        try:
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                tmp_file.write(audio_bytes)
                audio_path = tmp_file.name

            # Transcribe
            return self.transcribe_audio(audio_path)

        except Exception as e:
            logger.error(f"Error transcribing audio bytes: {str(e)}")
            return None
