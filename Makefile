sources = src tests

.PHONY: format
format:
	isort $(sources)
	black $(sources)

.PHONY: lint
lint:
	ruff check $(sources)
	isort $(sources) --check-only --df
	black $(sources) --check --diff

.PHONY: mypy
mypy:
	mypy $(sources) --config-file mypy.ini

.PHONY: all
all:
	make format
	make lint
	make mypy

.PHONY: test
test:
	pytest -W ignore::DeprecationWarning  tests