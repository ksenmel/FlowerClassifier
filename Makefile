VENV = .venv
ifeq ($(OS),Windows_NT)
	VENV_BIN_DIR = $(VENV)/Scripts
else
	VENV_BIN_DIR = $(VENV)/bin
endif
ACTIVATE = $(VENV_BIN_DIR)/activate

PYTHON = $(VENV_BIN_DIR)/python
PIP = $(VENV_BIN_DIR)/pip

BLACK = $(VENV_BIN_DIR)/black
MYPY = $(VENV_BIN_DIR)/mypy
PYTEST = $(VENV_BIN_DIR)/pytest

PY_SRCS = main.py $(wildcard **/*.py)


.PHONY: all
all: install-deps


# Run application

.PHONY: run
run: install-deps $(PY_SRCS)
	$(PYTHON) main.py


# Install dependencies for the application

.PHONY: install-deps
install-deps: $(ACTIVATE)

$(ACTIVATE): requirements.txt
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt


# Check application

.PHONY: check
check: check-formatting check-linting test

.PHONY: check-formatting
check-formatting: $(BLACK)
	$(BLACK) --check .

.PHONY: check-linting
check-linting: $(MYPY)
	$(MYPY) .

.PHONY: test
test: $(PYTEST)
	$(PYTHON) -m pytest tests


# Format application

.PHONY: format
format: $(BLACK)
	$(BLACK) .


# Install dependencies for check and format

$(BLACK): $(ACTIVATE)
	$(PIP) install black

$(MYPY): $(ACTIVATE)
	$(PIP) install mypy

$(PYTEST): $(ACTIVATE)
	$(PIP) install pytest


# Clean

.PHONY: clean
clean:
	rm -rf ./**/__pycache__

	rm -rf .mypy_cache
	rm -rf .pytest_cache

.PHONY: clean-full
clean-full: clean
	rm -rf $(VENV)
