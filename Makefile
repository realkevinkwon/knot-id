venv:
	python3 -m venv venv --upgrade-deps

requirements: venv requirements.txt
	./venv/bin/pip install -r requirements.txt

run: venv requirements
	./venv/bin/python3 src/models/train.py
	
.PHONY: requirements run