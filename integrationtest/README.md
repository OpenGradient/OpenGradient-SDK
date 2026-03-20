# Integration Tests

End-to-end tests that exercise the OpenGradient SDK against live services.

## Test Suites

### LLM (`llm/`)

Tests LLM chat and streaming chat via the x402 payment flow on Base Sepolia.

Each run creates a **fresh Ethereum account**, funds it with ETH (for gas) and OPG tokens from a funder wallet, approves Permit2, and then runs chat requests against a TEE-verified model.

**Requirements:**
- `PRIVATE_KEY` env var — private key of a funded wallet on Base Sepolia that holds OPG tokens.

```bash
make llm_integrationtest
```

### Agent (`agent/`)

Tests the on-chain agent inference flow.

```bash
make integrationtest
```

## CI

LLM integration tests run automatically on PRs and pushes to `main` via the **E2E Tests** GitHub Action (`.github/workflows/e2e.yml`). The `PRIVATE_KEY` secret must be configured in the repository settings.
