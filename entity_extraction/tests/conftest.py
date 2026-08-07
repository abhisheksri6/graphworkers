"""Dummy Celery env so `import entity_extraction_worker` (which instantiates Settings at module
import, like classification) works under pytest without a real broker. No network is made — the
broker URL is only wired into the Celery app object, never connected during these unit tests."""
import os

os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")
os.environ.setdefault("WORKER_RESULTS_URL", "http://localhost:8000/entity-extraction/worker-results")
os.environ.setdefault("RUNTIME_WORKER_RESULTS_URL", "http://localhost:8000/entity-extraction/runtime-worker-results")
