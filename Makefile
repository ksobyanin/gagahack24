.PHONY: *

MLVENV=.mlvenv
FLVENV=.flvenv

MLPYTHON = $(MLVENV)/bin/python3
FLPYTHON = $(FLVENV)/bin/python3

ml_venv:
	python3 -m venv $(MLVENV)

fl_venv:
	python3 -m venv $(FLVENV)

install_ml: ml_venv
	$(MLPYTHON) -m pip install -r ml_app/requirements.txt

install_fl: fl_venv
	$(FLPYTHON) -m pip install -r flask_app/requirements.txt

install_all: install_ml install_fl

run_flask:
	cd flask_app &&	../$(FLPYTHON) app.py

run_ml_back:
	cd ml_app && ../$(MLPYTHON) app.py

build:
	docker compose up -d --build