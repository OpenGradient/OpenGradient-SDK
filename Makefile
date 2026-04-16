# OpenGradient SDK Makefile

# Default model for testing (override with: make chat MODEL=google/gemini-3-pro-preview)
MODEL ?= google/gemini-3-pro-preview

# ============================================================================
# Development
# ============================================================================

install:
	uv sync --all-extras

build:
	uv build

publish:
	@echo "Current version:" $$(grep 'version = ' pyproject.toml | cut -d'"' -f2)
	rm -rf dist/*
	uv build
	uv publish

check:
	uv run ruff format .
	uv run mypy src
	uv run mypy examples

docs:
	uv run pdoc3 opengradient -o docs --template-dir ./templates --force

# ============================================================================
# Testing
# ============================================================================

test:
	uv run pytest tests/ -v

integrationtest:
	uv run python integrationtest/agent/test_agent.py
	uv run python integrationtest/workflow_models/test_workflow_models.py

llm_integrationtest:
	uv run python -m pytest integrationtest/llm/test_llm_chat.py -v

examples:
	@for f in $$(find examples -name '*.py' | sort); do \
		echo "Running $$f..."; \
		uv run python "$$f" || exit 1; \
	done
	@echo "All examples passed."

# ============================================================================
# CLI Examples (use MODEL=provider/model to change model)
# ============================================================================

infer:
	uv run python -m opengradient.cli infer \
		-m "hJD2Ja3akZFt1A2LT-D_1oxOCz_OtuGYw4V9eE1m39M" \
		--input '{"open_high_low_close": [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]}'

completion:
	uv run python -m opengradient.cli completion \
		--model $(MODEL) \
		--prompt "Hello, how are you?" \
		--max-tokens 50

chat:
	uv run python -m opengradient.cli chat \
		--model $(MODEL) \
		--messages '[{"role":"user","content":"Tell me a fun fact"}]' \
		--max-tokens 350

chat-stream:
	uv run python -m opengradient.cli chat \
		--model $(MODEL) \
		--messages '[{"role":"user","content":"Tell me a short story"}]' \
		--max-tokens 1250 \
		--stream

chat-tool:
	uv run python -m opengradient.cli chat \
		--model $(MODEL) \
		--messages '[{"role":"system","content":"You are a helpful assistant. Use tools when needed."},{"role":"user","content":"What'\''s the weather like in Dallas, Texas? Give me the temperature in fahrenheit."}]' \
		--tools '[{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather in a given location","parameters":{"type":"object","properties":{"city":{"type":"string"},"state":{"type":"string"},"unit":{"type":"string","enum":["fahrenheit","celsius"]}},"required":["city","state","unit"]}}}]' \
		--max-tokens 200

chat-stream-tool:
	uv run python -m opengradient.cli chat \
		--model $(MODEL) \
		--messages '[{"role":"system","content":"You are a helpful assistant. Use tools when needed."},{"role":"user","content":"What'\''s the weather like in Dallas, Texas? Give me the temperature in fahrenheit."}]' \
		--tools '[{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather in a given location","parameters":{"type":"object","properties":{"city":{"type":"string"},"state":{"type":"string"},"unit":{"type":"string","enum":["fahrenheit","celsius"]}},"required":["city","state","unit"]}}}]' \
		--max-tokens 200 \
		--stream

.PHONY: install build publish check docs test integrationtest llm_integrationtest examples \
	infer completion chat chat-stream chat-tool chat-stream-tool
