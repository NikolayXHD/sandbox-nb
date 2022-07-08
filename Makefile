jupyter:
	conda run --no-capture-output -p ./envs jupyter notebook --no-browser

jupyterlab:
	$(CONDA_RUN) jupyter lab --no-browser

lab: jupyterlab

clear:
	rm -rf ./.mypy_cache ./.pytest_cache
	find . -not \( -type d -name .storage -prune \) -type f -name "*.pyc" -print0 | xargs -r0 rm

format:
	$(CONDA_RUN) python -m brunette . \
--exclude '\.jupyter/' \
--single-quotes \
--target-version py38 \
--line-length 79

lint:
	$(CONDA_RUN) python -m flake8
	$(CONDA_RUN) python -m mypy notebooks tests

test:
	$(CONDA_RUN) python -m pytest -l tests

# input: make test-v
# result: pipenv run python -m pytest -lv tests
test-%:
	$(CONDA_RUN) python -m pytest -l$* tests

test-failed:
	$(CONDA_RUN) python -m pytest -l --last-failed tests

check: format lint test

# conda install mamba -n base -c conda-forge
env-create:
	mamba env create -p ./envs -f environment.yml

env-update:
	mamba env update -p ./envs -f environment.yml


SHELL := /usr/bin/bash

# Q: why `$(CONDA_RUN) some_command` instead of just
# `conda run --no-capture-output some_command`
# A: code executed via conda run does not exit on interrupt
# https://github.com/conda/conda/issues/11420
# CONDA_ACTIVATE implementation was borrowed from
# https://stackoverflow.com/a/71548453/6656775
.ONESHELL:
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate
CONDA_RUN = $(CONDA_ACTIVATE) ./envs;
