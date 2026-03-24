install:
	uv pip install -e ".[chroma,qdrant]"
test:
	PYTHONPATH=src python3 -m unittest -v

format:
	uv run black .