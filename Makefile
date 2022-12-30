venv:
	python3 -m venv venv --upgrade-deps

requirements: venv requirements.txt
	./venv/bin/pip install -r requirements.txt

train: venv requirements
	./venv/bin/python3 src/train.py

process: venv requirements
	./venv/bin/python3 src/process.py
	
.PHONY: requirements train process