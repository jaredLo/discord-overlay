SHELL := /bin/bash

ENV_NAME ?= discord-overlay

.PHONY: start
start:
	@if [ ! -f .installed ]; then \
		echo "Running 'make install' first..."; \
		$(MAKE) install; \
	fi; \
	conda run -n "$(ENV_NAME)" --no-capture-output python api/listener.py

.PHONY: install
install:
	@conda env list | awk '{print $$1}' | grep -qx "$(ENV_NAME)" || conda create -n "$(ENV_NAME)" python=3.9 -y
	@conda run -n "$(ENV_NAME)" --no-capture-output python -m pip install -r requirements.txt
	@touch .installed
