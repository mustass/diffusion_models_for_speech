.PHONY: style tests

check_dirs := src tests scripts
clean_dirs := wandb hf_cache .pytest_cache

style:
	black $(check_dirs)
	isort $(check_dirs)

lint:
	flake8 $(check_dirs)

tests:
	pytest -sv tests

clean:
	rm -rf $(clean_dirs)
