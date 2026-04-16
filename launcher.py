"""
TTM Ask - Launcher Service
A tiny HTTP server on port 5001 that starts Ollama and the main backend.
Started by Start TTM Ask.vbs and called by the HTML frontend.
"""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import subprocess
import os
import sys
import urllib.request
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable
APP_PY = os.path.join(BASE_DIR, 'app.py')
DISCOVER_PY = os.path.join(BASE_DIR, 'discover_online_docs.py')
OLLAMA_KEEP_ALIVE = os.getenv('OLLAMA_KEEP_ALIVE', '-1')
AUTO_ONLINE_DISCOVERY = os.getenv('TTM_ASK_AUTO_ONLINE_DISCOVERY', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
LOG_DIR = os.path.join(BASE_DIR, 'logs')
LAUNCHER_LOG = os.path.join(LOG_DIR, 'launcher.log')

# Windows flag: don't open a console window for spawned processes
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)

# Track background discovery process to avoid multiple parallel runs.
_discovery_proc = None


os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=LAUNCHER_LOG,
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def _check_url(url):
    try:
        urllib.request.urlopen(url, timeout=1)
        return True
    except Exception:
        return False


def is_ollama_running():
    return _check_url('http://localhost:11434/')


def is_backend_running():
    return _check_url('http://localhost:5000/health')


def start_services():
    global _discovery_proc
    started = []
    logger.info('Launcher start_services called')
    # Start Ollama if not already running
    if not is_ollama_running():
        try:
            ollama_env = os.environ.copy()
            ollama_env['OLLAMA_KEEP_ALIVE'] = OLLAMA_KEEP_ALIVE
            subprocess.Popen(
                ['ollama', 'serve'],
                creationflags=NO_WINDOW,
                cwd=BASE_DIR,
                env=ollama_env
            )
            started.append('ollama')
            logger.info('Started Ollama service')
        except FileNotFoundError:
            started.append('ollama_not_found')
            logger.warning('Ollama executable not found on PATH')

    # Start main Flask backend if not already running
    if not is_backend_running():
        subprocess.Popen(
            [PYTHON_EXE, APP_PY],
            creationflags=NO_WINDOW,
            cwd=BASE_DIR
        )
        started.append('backend')
        logger.info('Started backend process with %s', PYTHON_EXE)

    # Optional online discovery at startup.
    # Disabled by default so the indexed standards set stays stable across launches.
    if AUTO_ONLINE_DISCOVERY and os.path.exists(DISCOVER_PY):
        if _discovery_proc is None or _discovery_proc.poll() is not None:
            _discovery_proc = subprocess.Popen(
                [
                    PYTHON_EXE,
                    DISCOVER_PY,
                    '--download',
                    '--min-score', '2',
                    '--max-download', '12',
                    '--report', os.path.join(BASE_DIR, 'online_discovery_report.json')
                ],
                creationflags=NO_WINDOW,
                cwd=BASE_DIR
            )
            started.append('online_discovery')
            logger.info('Started online discovery process')
        else:
            started.append('online_discovery_running')
    elif AUTO_ONLINE_DISCOVERY:
        started.append('online_discovery_script_missing')
        logger.warning('Online discovery enabled but script missing: %s', DISCOVER_PY)

    return started


class LauncherHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.info('HTTP %s', format % args)

    def _cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self._cors_headers()
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError) as exc:
            logger.warning('Client disconnected while writing response: %s', exc)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == '/status':
            self._send_json({
                'launcher': True,
                'ollama': is_ollama_running(),
                'backend': is_backend_running()
            })
        elif self.path == '/start':
            started = start_services()
            self._send_json({'started': started})
        else:
            self._send_json({'error': 'Not found'}, 404)

    def do_POST(self):
        if self.path == '/start':
            started = start_services()
            self._send_json({'started': started})
        else:
            self._send_json({'error': 'Not found'}, 404)


if __name__ == '__main__':
    try:
        server = ReusableThreadingHTTPServer(('localhost', 5001), LauncherHandler)
        logger.info('TTM Ask launcher ready on http://localhost:5001')
        print('TTM Ask launcher ready on http://localhost:5001')
        server.serve_forever()
    except Exception as exc:
        logger.exception('Launcher failed to start: %s', exc)
        raise
