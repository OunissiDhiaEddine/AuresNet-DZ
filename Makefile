.PHONY: format lint test prepare-data train

format:
	black src tests
	ruff check --fix src tests

lint:
	ruff check src tests

test:
	pytest -q

prepare-data:
	python scripts/prepare_aures_data.py

train:
	python -m auresnet_dz.train.train
