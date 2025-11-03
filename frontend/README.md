# Azazel AI - Modern Web Interface

Beautiful ChatGPT-like interface built with HTML, Tailwind CSS, and vanilla JavaScript.

## Features

✅ **Modern Design** - Clean, responsive ChatGPT-style interface
✅ **Real-time Streaming** - See responses as they're generated
✅ **Markdown Support** - Full markdown rendering with code highlighting
✅ **Syntax Highlighting** - Beautiful code blocks with highlight.js
✅ **Document Upload** - Support for PDF, DOCX, images, and more
✅ **Web Search** - Integrated web search capabilities
✅ **Code Execution** - Run Python code safely
✅ **Mobile Responsive** - Works perfectly on all devices
✅ **No Framework** - Pure HTML/CSS/JS, no build step required

## Quick Start

### Local Development

1. Start the frontend server:
```bash
cd frontend
python server.py
```

2. Open your browser:
```
http://localhost:8080
```

3. Enter your OpenAI API key in the sidebar

4. Start chatting!

### With Docker

```bash
docker build -t azazel-frontend .
docker run -p 8080:8080 azazel-frontend
```

### With Full Stack

```bash
# From project root
docker-compose up
```

This starts:
- Frontend: http://localhost:8080
- API: http://localhost:8000

## Deployment

### Railway (Recommended)

1. **Deploy Backend**:
   - Create new project on Railway
   - Connect your GitHub repo
   - Set root directory to `/api`
   - Add environment variables (API keys)
   - Deploy

2. **Deploy Frontend**:
   - Create another service in same project
   - Set root directory to `/frontend`
   - Set start command: `python server.py`
   - Deploy

3. **Update API URL**:
   - In `frontend/static/js/config.js`, update `API_URL` with your Railway backend URL

### Render

1. **Backend**:
   - Create Web Service
   - Build: `pip install -r api/requirements.txt`
   - Start: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

2. **Frontend**:
   - Create Static Site
   - Build: (none)
   - Publish: `frontend`

### Vercel (Frontend Only)

```bash
cd frontend
vercel
```

Then update `API_URL` in config.js with your backend URL.

## Configuration

### API Endpoint

Edit `frontend/static/js/config.js`:

```javascript
const CONFIG = {
    API_URL: 'https://your-api-url.com',
    // ... other settings
};
```

### Features

Toggle features in the sidebar:
- **Web Search** - Search the internet for current information
- **Code Execution** - Execute Python code
- **Document Mode** - Upload and query documents

## File Structure

```
frontend/
├── index.html           # Main HTML file
├── server.py           # Simple Python HTTP server
├── Dockerfile          # Docker configuration
├── static/
│   ├── js/
│   │   ├── config.js   # Configuration
│   │   ├── api-client.js  # API client
│   │   └── app.js      # Main application logic
│   ├── css/            # Custom styles (if any)
│   └── images/         # Images and assets
└── README.md
```

## Technologies Used

- **Tailwind CSS** - Utility-first CSS framework
- **Marked.js** - Markdown parser
- **Highlight.js** - Syntax highlighting
- **Fetch API** - Modern HTTP requests
- **Server-Sent Events** - Real-time streaming

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

## Performance

- **No Build Step** - Instant loading
- **Minimal Dependencies** - Only CDN libraries
- **Streaming Responses** - Real-time updates
- **LocalStorage** - Persistent settings and chat history

## Security

- API keys stored in browser LocalStorage only
- No server-side key storage
- CORS enabled for API communication

## Customization

### Colors

Edit Tailwind classes in `index.html`:
- Primary: `bg-blue-600`, `text-blue-600`
- Background: `bg-gray-50`
- Messages: `bg-white`

### Styling

Add custom CSS in `<style>` tag or create `static/css/custom.css`

## License

MIT