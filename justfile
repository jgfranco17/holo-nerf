# List out available commands
default:
	@just --list

# Execute installation
setup:
	@echo "Setting up project."
	poetry install
	poetry shell
	@echo "Project setup complete!"

# Launch API in debug mode
run-debug:
	@echo "Running main app..."
	@python3 app.py --port 8080 --debug

# Launch API in production mode
run-prod:
	@echo "Running main app..."
	@python3 app.py --port 8080

# Build Docker image
build-docker:
	@echo "Building docker image..."
	docker build -t fizzbuzz-api:latest -f ./Dockerfile .
	@echo "Docker image built successfully!"

# Clean unused files
clean:
	-@find ./ -name '*.pyc' -exec rm -f {} \;
	-@find ./ -name '__pycache__' -exec rm -rf {} \;
	-@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	-@find ./ -name '*~' -exec rm -f {} \;
	-@rm -rf .pytest_cache
	-@rm -rf .cache
	-@rm -rf .mypy_cache
	-@rm -rf build
	-@rm -rf dist
	-@rm -rf *.egg-info
	-@rm -rf htmlcov
	-@rm -rf .tox/
	-@rm -rf docs/_build
	-@rm -rf .venv
	@echo "Cleaned out unused files and directories!"

# Run PyTest unit tests
pytest *args:
	@echo "Running unittest suite..."
	poetry run pytest {{ args }}
	-@find ./ -name '__pycache__' -exec rm -rf {} \;
	-@rm -rf .pytest_cache
	@echo "Cleaned up test environment"

coverage:
    @poetry run coverage run -m pytest
    @poetry run coverage report

# Run Behave feature tests
behave:
	@echo "Running feature test suite..."
	poetry run behave ./tests/features/feature_tests/api_features

# Run load tests
locust:
    @echo "Running load testing"
    locust -f ./tests/load-test/locustfile.py

smoke-tests:
	@echo "Running smoke test suite..."
	python3 tests/smoke-tests/smoketests.py

# Start Compose with load-balancer
docker-up:
    docker compose -f docker/docker-compose.yml up

# Stop all Compose containers and delete images created
docker-down:
    docker compose -f docker/docker-compose.yml down
    docker rmi $(docker images | grep "fizzbuzz-app" | awk "{print \$3}")
