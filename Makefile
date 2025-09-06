SHELL := /bin/bash
.ONESHELL:

# Config
ENV_NAME ?= discord-overlay
PORT ?= 8201
CONDA ?= conda
PY ?= python
NVM_DIR ?= $(HOME)/.nvm

.PHONY: help env-file install update dev run-api desktop-dev desktop-build desktop-prereqs desktop-icon-placeholder stop logs-clean clean api tauri

help:
	@echo "Targets:"
	@echo "  make dev             # Start API (spawns listener) + Tauri; stops API on exit"
	@echo "  make run-api         # Run FastAPI in foreground (uses conda if not active)"
	@echo "  make desktop-dev     # Launch Tauri dev"
	@echo "  make desktop-build   # Build Tauri installers"
	@echo "  make desktop-prereqs # Check/install Rust and Node deps"
	@echo "  make desktop-icon-placeholder # Create placeholder icon.png for Tauri"
	@echo "  make install         # Install Python and Node dependencies"
	@echo "  make stop            # Stop background API from make dev"
	@echo "  make logs-clean      # Remove logs/ and tmp/"
	@echo "  make clean           # Remove logs, tmp, node/tauri build caches"

env-file:
	@[ -f .env ] || (cp .env-sample .env && echo "[env] Copied .env-sample to .env") || true

install: env-file ## Install Python and Node deps
	@set -euo pipefail; \
	$(CONDA) run -n $(ENV_NAME) $(PY) -m pip install -U pip; \
	$(CONDA) run -n $(ENV_NAME) pip install -r requirements.txt; \
	cd desktop && npm install

update: install ## Alias to install

dev: env-file ## Run API + Tauri together
	@set -euo pipefail; \
	mkdir -p logs tmp; \
	echo "[dev] Starting API on :$(PORT) ..."; \
	$(CONDA) run -n $(ENV_NAME) $(PY) -c 'import uvicorn' 2>/dev/null || { echo "[dev] uvicorn not found in env $(ENV_NAME). Run: make install"; exit 1; }; \
	$(CONDA) run -n $(ENV_NAME) $(PY) -m uvicorn server.main:app --port $(PORT) > logs/api.log 2>&1 & \
	API_PID=$$!; \
	echo $$API_PID > tmp/api.pid; \
	echo "[dev] API pid=$$API_PID (logs/api.log)"; \
	echo "[dev] Starting Tauri dev ..."; \
	export NVM_DIR="$(NVM_DIR)"; \
	[ -s "$$NVM_DIR/nvm.sh" ] && . "$$NVM_DIR/nvm.sh"; \
	nvm use v24 >/dev/null; \
	cd desktop; \
	npm install; \
	trap 'echo "[dev] Stopping API $$API_PID"; kill $$API_PID 2>/dev/null || true; wait $$API_PID 2>/dev/null || true' EXIT; \
	npm run tauri:dev

run-api: env-file ## Run only FastAPI in foreground
	@set -euo pipefail; \
	if [ "$$CONDA_DEFAULT_ENV" = "$(ENV_NAME)" ]; then \
	  echo "[api] Using active env $(ENV_NAME)"; \
	  $(PY) -c 'import uvicorn' 2>/dev/null || { echo "[api] uvicorn not found in active env. Run: make install"; exit 1; }; \
	  $(PY) -m uvicorn server.main:app --reload --port $(PORT); \
	else \
	  echo "[api] Using conda run (env $(ENV_NAME))"; \
	  $(CONDA) run -n $(ENV_NAME) $(PY) -c 'import uvicorn' 2>/dev/null || { echo "[api] uvicorn not found in env $(ENV_NAME). Run: make install"; exit 1; }; \
	  $(CONDA) run -n $(ENV_NAME) $(PY) -m uvicorn server.main:app --reload --port $(PORT); \
	fi

desktop-dev: ## Run only Tauri dev (expects API running)
	@set -euo pipefail; \
	export NVM_DIR="$(NVM_DIR)"; \
	[ -s "$$NVM_DIR/nvm.sh" ] && . "$$NVM_DIR/nvm.sh"; \
	nvm use v24 >/dev/null; \
	cd desktop; \
	npm install; \
	npm run tauri:dev

desktop-build:
	@set -euo pipefail; \
	export NVM_DIR="$(NVM_DIR)"; \
	[ -s "$$NVM_DIR/nvm.sh" ] && . "$$NVM_DIR/nvm.sh"; \
	nvm use v24 >/dev/null; \
	cd desktop && npm install && npm run tauri:build

desktop-prereqs:
	@echo "[desktop] Checking prerequisites for Tauri (Rust, Xcode CLT on macOS, Node)"; \
	if [ "`uname`" = "Darwin" ]; then \
	  if ! xcode-select -p >/dev/null 2>&1; then \
	    echo "[desktop] Xcode Command Line Tools not found. Run: xcode-select --install"; \
	  else echo "[desktop] Xcode Command Line Tools present"; fi; \
	fi; \
	if ! command -v cargo >/dev/null 2>&1; then \
	  echo "[desktop] Rust not found. Install rustup and restart shell."; \
	else echo "[desktop] Rust is installed: $$(cargo --version)"; fi; \
	cd desktop && npm install

desktop-icon-placeholder:
	@echo "[desktop] Writing placeholder icon to desktop/src-tauri/icons/icon.png"; \
	mkdir -p desktop/src-tauri/icons; \
	node -e "require('fs').writeFileSync('desktop/src-tauri/icons/icon.png', Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9W2n7wAAAABJRU5ErkJggg==','base64')); console.log('[desktop] Placeholder icon created');"

stop: ## Stop background API started by `make dev`
	@set -euo pipefail; \
	if [ -f tmp/api.pid ]; then \
	  PID=$$(cat tmp/api.pid); \
	  echo "[stop] Killing API $$PID"; \
	  kill $$PID 2>/dev/null || true; \
	  rm -f tmp/api.pid; \
	else \
	  echo "[stop] No tmp/api.pid found"; \
	fi

logs-clean:
	@rm -rf logs tmp; \
	 echo "[clean] Removed logs/ and tmp/"

clean: logs-clean ## Remove build caches and logs
	rm -rf desktop/dist desktop/node_modules desktop/src-tauri/target

# Back-compat aliases
api: run-api
tauri: desktop-dev
