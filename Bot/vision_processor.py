# vision_processor.py
import os
import base64
import logging
import tempfile
from openai import OpenAI
from typing import List, Dict, Any
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class VisionProcessor:
    """Process images and integrate with GPT-4o's vision capabilities"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.image_store = {}
        self.image_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploaded_images")
        os.makedirs(self.image_folder, exist_ok=True)

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            return ""

    def process_image(self, uploaded_file, session_id: str) -> bool:
        """Store uploaded image for a session"""
        try:
            if session_id not in self.image_store:
                self.image_store[session_id] = []

            # Handle both FastAPI (filename) and Streamlit (name)
            file_name = getattr(uploaded_file, 'filename', getattr(uploaded_file, 'name', 'image.jpg'))
            file_extension = file_name.split('.')[-1]
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}.{file_extension}"
            permanent_path = os.path.join(self.image_folder, filename)

            # Handle both FastAPI (read()) and Streamlit (getvalue())
            file_content = uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file.getvalue()

            with open(permanent_path, "wb") as f:
                f.write(file_content if isinstance(file_content, bytes) else file_content)

            self.image_store[session_id].append(permanent_path)
            return True
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return False

    def analyze_images(self, session_id: str, query: str) -> List[Dict[str, Any]]:
        """Analyze images using GPT-4o's vision model"""
        if session_id not in self.image_store or not self.image_store[session_id]:
            return []

        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(img_path)}"}} 
                      for img_path in self.image_store[session_id]]
                ]
            }]

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1000
            )
            
            return [{
                "type": "image_analysis",
                "content": response.choices[0].message.content,
                "images": self.image_store[session_id]
            }]
        except Exception as e:
            error_message = str(e)
            if "quota" in error_message.lower() or "rate limit" in error_message.lower():
                logger.error(f"Vision API quota exceeded for your API key. Please check your OpenAI account limits or try again later.")
            else:
                logger.error(f"Vision API error: {error_message}")
            return []
