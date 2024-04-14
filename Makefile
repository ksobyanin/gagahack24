.PHONY: *

VENV=.venv

PYTHON = $(VENV)/bin/python3

venv:
	python3 -m venv $(VENV)

fl_venv:
	python3 -m venv $(FLVENV)

install: venv
	$(PYTHON) -m pip install -r flask_app/requirements.txt

run:
	cd flask_app &&	../$(MLPYTHON) app.py

build:
	docker compose up -d --build