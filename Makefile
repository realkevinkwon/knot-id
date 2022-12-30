venv:
	python3 -m venv venv --upgrade-deps

requirements: venv requirements.txt
	./venv/bin/pip install -r requirements.txt

process: venv requirements
	./venv/bin/python3 src/process.py
	
train: venv requirements
	./venv/bin/python3 src/train.py

predict: venv requirements
	./venv/bin/python3 src/predict.py

.PHONY: requirements process train predict