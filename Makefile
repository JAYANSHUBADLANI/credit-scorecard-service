PYTHON ?= python3
API_HOST ?= 127.0.0.1
API_PORT ?= 8000

.PHONY: help install install-dev train demo api monitor monitor-once stream dashboard test noise compose-check clean clean-data

help:
	@echo "make install       install the pinned runtime dependencies"
	@echo "make install-dev   also install the test tooling, needed for make test"
	@echo "make demo          the whole system end to end, locally: train, serve, stream, monitor, report"
	@echo "make train         fit the scorecard and write the artifact and the reference distributions"
	@echo "make api           run the scoring API on $(API_HOST):$(API_PORT)"
	@echo "make stream        post the simulated scoring stream at a running API"
	@echo "make monitor       run the drift monitor continuously"
	@echo "make monitor-once  process one window and exit"
	@echo "make dashboard     open the monitoring dashboard on 8501"
	@echo "make test          run the test suite"
	@echo "make noise         measure the stability index noise floor by window size"
	@echo "make compose-check validate the compose file without a Docker daemon"
	@echo "make clean         remove caches"
	@echo "make clean-data    also remove the monitoring database and generated reports"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

# The single documented entrypoint. Everything the README quotes comes from this target.
demo:
	$(PYTHON) scripts/run_end_to_end.py --host $(API_HOST) --port $(API_PORT)

train:
	$(PYTHON) -m src.train

api:
	$(PYTHON) -m uvicorn src.api:app --host $(API_HOST) --port $(API_PORT) --reload

stream:
	$(PYTHON) -m src.stream --api-url http://$(API_HOST):$(API_PORT)

monitor:
	$(PYTHON) -m src.monitor --loop --interval 30

monitor-once:
	$(PYTHON) -m src.monitor --flush --export

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py --server.port 8501

test:
	$(PYTHON) -m pytest -q

noise:
	$(PYTHON) scripts/window_size_noise.py

compose-check:
	$(PYTHON) scripts/validate_compose.py

clean:
	find . -name __pycache__ -type d -exec rm -rf {} +
	rm -rf .pytest_cache

clean-data:
	rm -f data/monitoring.db data/monitoring.db-wal data/monitoring.db-shm
	rm -f reports/drift_metrics.csv reports/window_summary.csv reports/alerts.csv
