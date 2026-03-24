install:
	uv pip install -e .
test:
	PYTHONPATH=src python3 -m unittest -v

format:
	uv run black .