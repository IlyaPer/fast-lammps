.PHONY: install test lint clean

install:
# 	pip install -e .
# 	pip install -r requirements-dev.txt  # для разработки 

test:
	python -m unittest tests/test_extractor.py

lint:
	flake8 src/ tests/
	black --check src/ tests/

clean:
# 	find . -type d -name "__pycache__" -exec rm -rf {} +
# 	rm -rf .pytest_cache

clean_records:
	rm -rf records/* logs/*

test_launch: clean_records
	python main.py -k 4 -i 4000 -m 500