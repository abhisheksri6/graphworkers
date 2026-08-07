"""Dummy Celery env so `import entity_canonicalization_worker` (Settings at module import) works under
pytest without a real broker. No network is made — the broker URL only wires the Celery app object."""
import os

os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")
os.environ.setdefault("WORKER_RESULTS_URL", "http://localhost:8000/entity-canonicalization/worker-results")
