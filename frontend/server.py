"""
Simple HTTP server for Azazel AI frontend
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

if __name__ == '__main__':
    # Change to frontend directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    port = int(os.getenv('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), CORSRequestHandler)

    print(f'🚀 Azazel AI Frontend running on http://0.0.0.0:{port}')
    print(f'📝 Open http://localhost:{port} in your browser')

    server.serve_forever()