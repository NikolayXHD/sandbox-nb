jupyter:
	conda run -p ./envs jupyter notebook

clear:
	rm -rf ./.mypy_cache ./.pytest_cache
	find . -not \( -type d -name .storage -prune \) -type f -name "*.pyc" -print0 | xargs -r0 rm

# install brunette into system python3 executable:
# > python3 -m pip install brunette==0.2.0 black
format:
	python3 -m brunette . --exclude '\.jupyter/' \
      --single-quotes \
      --target-version py38 \
      --line-length 79

lint:
	conda run -p ./envs python -m flake8
	conda run -p ./envs python -m mypy notebooks tests

test:
	conda run -p ./envs python -m pytest -l tests

# input: make test-v
# result: pipenv run python -m pytest -lv tests
test-%:
	conda run -p ./envs python -m pytest -l$* tests

test-failed:
	conda run -p ./envs python -m pytest -l --last-failed tests

check: format lint test

env-create:
	conda env create -p ./envs -f environment.yml

env-update:
	conda env update -p ./envs -f environment.yml


SHELL := /usr/bin/bash
