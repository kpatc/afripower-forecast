.PHONY: setup data train api dashboard test clean

PYTHON  := venv/bin/python
PIP     := venv/bin/pip

setup:
	python -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	cp -n .env.example .env || true
	@echo "Setup complete. Edit .env to add your API keys."

data:
	$(PYTHON) src/data/fetch_data.py

train:
	$(PYTHON) -m src.models.baseline
	$(PYTHON) -m src.models.ml_models
	$(PYTHON) -m src.models.deep_learning

api:
	venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	venv/bin/streamlit run dashboard/app.py

test:
	venv/bin/pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	@echo "Cleaned."
