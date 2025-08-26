SHELL := /bin/bash

ENV_NAME ?= discord-overlay

.PHONY: start
start:
	@eval "$$(conda shell.bash hook)" && conda activate "$(ENV_NAME)" && python listener.py


