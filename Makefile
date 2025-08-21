.PHONY: help install dev-install test clean run-basic run-all format lint code-quality

help:  ## Show this help message
	@echo "Numerical Optimization Project 3 - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install project dependencies
	uv sync

dev-install:  ## Install development dependencies
	uv sync --extra dev

test:  ## Run tests
	uv run python -m pytest tests/ -v

format:  ## Format code with black
	uv run black src/ --line-length 88

lint:  ## Lint code with flake8
	uv run flake8 src/ --max-line-length 88 --ignore E203,W503

format-check:  ## Check if code is formatted correctly
	uv run black --check src/ --line-length 88

code-quality: format lint  ## Run both formatting and linting

clean:  ## Clean up generated files
	rm -rf results/plots/*
	rm -rf results/data/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

run-basic:  ## Run basic sine function test
	uv run python src/experiments/test_sine_problem.py

run-task1:  ## Run Task 1 algorithms test
	uv run python src/experiments/test_task1_algorithms.py

run-task2:  ## Run Task 2 comprehensive experiments
	uv run python src/experiments/test_task2_experiments.py

run-task3:  ## Run Task 3 pre-conditioning analysis
	uv run python src/experiments/task3_preconditioning.py

run-task4:  ## Run Task 4 pre-conditioned runs and comparisons
	uv run python src/experiments/task4_preconditioned_runs.py

generate-images:  ## Generate all images and save data for Tasks 1 & 2
	uv run python src/experiments/generate_all_images.py

run-all:  ## Run all experiments
	uv run python src/experiments/run_all.py

setup: install  ## Setup project (install dependencies)
	@echo "Project setup complete!"
	@echo "Run 'make run-basic' to test your environment"
