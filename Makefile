.PHONY: setup data train evaluate serve test docker-build docker-run all

setup:
	pip install -r requirements.txt

data:
	cd data && python getdata.py && cd ..
	python -c "from src import data_preparation; df=data_preparation.load_data(); data_preparation.validate_data(df); data_preparation.eda(df)"

train:
	python -c "from src import train; train.train()"

evaluate:
	python -c "from src import evaluate; evaluate.evaluate()"

serve:
	uvicorn src.app:app --host 0.0.0.0 --port 8000

test:
	pytest -v

docker-build:
	docker build -t student-marks-prediction:latest .

docker-run:
	docker run -p 8000:8000 student-marks-prediction:latest

all: setup data train evaluate test
	@echo "All done"
