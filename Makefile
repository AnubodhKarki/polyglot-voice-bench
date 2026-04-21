.PHONY: install test run-v0 clean

install:
	uv sync

test:
	uv run pytest tests/ -v

run-v0:
	uv run python scripts/run_v0.py

clean:
	rm -rf results/transcripts/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
