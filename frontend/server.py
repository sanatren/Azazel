"""
Simple HTTP server for Azazel AI frontend with environment variable injection
"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        # Inject backend URL into index.html
        if self.path == '/' or self.path == '/index.html':
            try:
                with open('index.html', 'r', encoding='utf-8') as f:
                    content = f.read()

                # Inject backend URL from environment variable
                backend_url = os.getenv('BACKEND_URL', 'http://localhost:8000')
                injection = f'<script>window.BACKEND_URL = "{backend_url}";</script>'
                content = content.replace('</head>', f'{injection}\n</head>')

                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Content-Length', len(content.encode()))
                self.end_headers()
                self.wfile.write(content.encode())
                return
            except Exception as e:
                print(f"Error serving index.html: {e}")

        # Default behavior for other files
        super().do_GET()

if __name__ == '__main__':
    # Change to frontend directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    port = int(os.getenv('PORT', 8080))
    backend_url = os.getenv('BACKEND_URL', 'http://localhost:8000')

    server = HTTPServer(('0.0.0.0', port), CORSRequestHandler)

    print(f'🚀 Azazel AI Frontend running on http://0.0.0.0:{port}')
    print(f'🔗 Backend API: {backend_url}')
    print(f'📝 Open http://localhost:{port} in your browser')

    server.serve_forever()