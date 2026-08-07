"""Dummy Celery env so `import kg_export_worker` (Settings at module import) works under pytest."""
import os

os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")
os.environ.setdefault("WORKER_RESULTS_URL", "http://localhost:8000/kg-export/worker-results")
