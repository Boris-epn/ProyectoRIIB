"""Un proceso comparte el corpus, el índice y la evaluación en memoria."""
import os

bind = f"0.0.0.0:{int(os.environ.get('PORT', '5000'))}"
workers = 1
worker_class = "gthread"
threads = 4
timeout = 300
accesslog = "-"
errorlog = "-"
preload_app = False
