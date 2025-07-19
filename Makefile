# Makefile

.PHONY: train test run lint format clean

# Train the model
train:
	python vehicle_insurance_fraud_detection/modeling/train.py

# Run tests
test:
	pytest tests/

# Run the Streamlit app
run:
	streamlit run app/app.py

# Lint the code
lint:
	flake8 vehicle_insurance_fraud_detection

# Format with Black and isort
format:
	black vehicle_insurance_fraud_detection
	isort vehicle_insurance_fraud_detection

# Clean __pycache__ files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +


