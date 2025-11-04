# 🤖 Azazel AI[https://brave-comfort-production-f62d.up.railway.app/] - Advanced AI Assistant

Modern, full-featured AI assistant with beautiful ChatGPT-like interface, document processing, web search, and code execution.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)

## ✨ Features

### 🎯 Core Capabilities
- **💬 Intelligent Chat** - Natural conversations powered by OpenAI GPT-4
- **🌐 Web Search** - Real-time information from Google with intelligent query optimization
- **📄 Document Processing** - Upload and query PDFs, DOCX, Excel, PowerPoint
- **🖼️ Vision Analysis** - Analyze and discuss images using GPT-4o-mini vision
- **💻 Code Execution** - Safe Python code generation and execution
- **🎭 Personality Modes** - Customizable AI personalities and tones
- **🌍 Multi-language** - Support for multiple languages
- **🎤 Speech Integration** - Voice input using OpenAI Whisper

### 🚀 Technical Features
- **Real-time Streaming** - See responses as they're generated
- **RAG (Retrieval-Augmented Generation)** - Context-aware responses from your documents
- **Session Management** - Persistent chat history with Supabase
- **Sentiment Analysis** - Emotion detection and adaptive responses
- **Model Switching** - GPT-4, GPT-4 Turbo, or GPT-4o-mini based on your needs
- **Modern UI** - Beautiful ChatGPT-like interface with Tailwind CSS
- **Mobile Responsive** - Works perfectly on all devices

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Frontend (HTML/CSS/JS)                    │
│  • Modern ChatGPT-like UI                               │
│  • Tailwind CSS styling                                 │
│  • Real-time streaming                                  │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/SSE
┌────────────────────┴────────────────────────────────────┐
│                  Backend (FastAPI)                       │
│  • RESTful APIs                                         │
│  • Streaming responses                                  │
│  • Document processing                                  │
│  • Web search integration                               │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────┴────────┐      ┌────────┴────────┐
│   OpenAI API   │      │  Google Search   │
│   • GPT-4      │      │  • Custom CSE    │
│   • Vision     │      │  • Wikipedia     │
│   • Whisper    │      │                  │
└────────────────┘      └──────────────────┘
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd Azazel

# Create environment file
cat > .env << EOF
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
EOF

# Start with Docker Compose
docker-compose up -d

# Access the app
# Frontend: http://localhost:8080
# API Docs: http://localhost:8000/docs
```

### Option 2: Local Development

**Backend:**
```bash
cd api
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
python server.py
# Visit http://localhost:8080
```

## 🌐 Deploy to Railway

See [DEPLOY.md](DEPLOY.md) for complete deployment instructions.

**Quick Steps:**
1. Push to GitHub
2. Connect to Railway
3. Deploy backend + frontend services
4. Add environment variables
5. Done! 🎉

## 🔧 Configuration

### Environment Variables

```env
# Required for web search
GOOGLE_API_KEY=AIzaSy...
GOOGLE_CSE_ID=0123456789...

# Optional for chat history
SUPABASE_URL=https://...
SUPABASE_KEY=eyJhbGci...
```

### Get API Keys

1. **OpenAI** (Users provide their own):
   - https://platform.openai.com/api-keys

2. **Google Search**:
   - API Key: https://console.cloud.google.com/apis/credentials
   - CSE ID: https://programmablesearchengine.google.com/

3. **Supabase** (Optional):
   - https://supabase.com/dashboard

## 📚 API Endpoints

- `POST /api/chat/message` - Send a message
- `POST /api/chat/stream` - Stream response
- `POST /api/chat/code` - Execute code
- `POST /api/documents/upload` - Upload file
- `POST /api/search/web` - Web search
- `GET /health` - Health check

Full API docs: `http://localhost:8000/docs`

## 🎨 Why Azazel?

**Supercharging AI Models** - Azazel elevates GPT-4 and legacy models to the next level by integrating:
- **RAG** - Chat with your documents
- **Vision** - Analyze images and screenshots
- **Code Execution** - Run Python code safely
- **Web Search** - Access real-time information
- **Multi-modal** - Handle text, images, code, and documents

All in one beautiful, easy-to-use interface without the heavyweight price tag!

## 🛠️ Built With

- **FastAPI** - Modern, fast Python web framework
- **LangChain** - AI orchestration and RAG
- **OpenAI** - GPT-4, Vision, Whisper
- **Tailwind CSS** - Beautiful, responsive UI
- **Docker** - Containerization
- **Railway** - Easy deployment

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- FastAPI for amazing web framework
- Tailwind CSS for beautiful styling
- LangChain for AI orchestration
- Railway for easy deployment

---

<div align="center">

**Built with ❤️ for the AI community**

[⭐ Star on GitHub](https://github.com/your-repo) • [🐛 Report Bug](https://github.com/your-repo/issues) • [✨ Request Feature](https://github.com/your-repo/issues)

</div>







