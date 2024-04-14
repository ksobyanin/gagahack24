.PHONY: *

VENV=.mlvenv

PYTHON = $(VENV)/bin/python3

venv:
	python3 -m venv $(VENV)

install: venv
	$(PYTHON) -m pip install -r flask_app/requirements.txt

run:
	cd flask_app &&	../$(PYTHON) app.py

build:
	docker compose up -d --build

compose_stop:
	docker compose stop

load_weights:
	wget -O flask_app/best_text_boxes.pt https://www.dropbox.com/scl/fi/xndg5depchdrtlgl0u8c0/best_text_boxes.pt?rlkey=ilk57aua6eh9gb8mkr1o5bk4u&dl=0
	wget -O flask_app/best_class.pt https://www.dropbox.com/scl/fi/hx5uvez2pcmo340m4cldq/best_class.pt?rlkey=axv3ucb2xdpv3vbfqstwwxn18&dl=0