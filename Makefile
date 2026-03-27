.PHONY: format lint test train

format:
	black src tests
	ruff check --fix src tests

lint:
	ruff check src tests

test:
	pytest -q

train:
	python -m auresnet_dz.train.train
