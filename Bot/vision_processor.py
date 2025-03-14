# vision_processor.py
import os
import base64
import streamlit as st
import tempfile
from openai import OpenAI
from typing import List, Dict, Any
import uuid
from datetime import datetime

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
            st.error(f"Error encoding image: {str(e)}")
            return ""

    def process_image(self, uploaded_file, session_id: str) -> bool:
        """Store uploaded image for a session"""
        try:
            if session_id not in self.image_store:
                self.image_store[session_id] = []

            file_extension = uploaded_file.name.split('.')[-1]
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}.{file_extension}"
            permanent_path = os.path.join(self.image_folder, filename)
            
            with open(permanent_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            self.image_store[session_id].append(permanent_path)
            return True
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return False

    def analyze_images(self, session_id: str, query: str) -> List[Dict[str, Any]]:
        """Analyze images using GPT-4o's vision model"""
        if session_id not in self.image_store or not self.image_store[session_id]:
            return []

        try:
            # Enhanced prompt to explicitly instruct the model to describe the image
            enhanced_query = f"""Please analyze and describe the following image in detail.
            
Original query: "{query}"

Instructions:
1. Describe what you see in the image with specific details
2. Mention colors, objects, people, clothing, backgrounds, and any notable features
3. Provide a comprehensive description that directly answers the query
4. Never say you cannot see or access the image - you CAN see the image
5. If asked about something not in the image, clarify what IS in the image instead

Always respond as if you can see the image, because you CAN see the image that was uploaded.
"""

            messages = [{
                "role": "system",
                "content": "You are a vision-enabled assistant that can see and describe images in detail. You should ALWAYS provide detailed descriptions of images when asked about them."
            }, {
                "role": "user",
                "content": [
                    {"type": "text", "text": enhanced_query},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(img_path)}"}} 
                      for img_path in self.image_store[session_id]]
                ]
            }]

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000
            )
            
            return [{
                "type": "image_analysis",
                "content": response.choices[0].message.content,
                "images": self.image_store[session_id]
            }]
        except Exception as e:
            st.error(f"Vision API error: {str(e)}")
            return []