import os

from waitress import serve

from app import app, APP_HOST, APP_PORT


if __name__ == '__main__':
    threads = int(os.getenv('TTM_ASK_THREADS', '8'))
    serve(app, host=APP_HOST, port=APP_PORT, threads=threads)