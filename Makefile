.PHONY: install clean-data train serve build-pipeline

install:
	pip install -r requirements.txt

clean-data:
	python pipeline/clean_data.py

train:
	python models/train_all.py

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

build-pipeline: install clean-data train serve
