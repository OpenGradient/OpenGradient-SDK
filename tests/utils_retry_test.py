"""Tests for client/_utils.py — get_abi, get_bin, and run_with_retry."""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from opengradient.client._utils import get_abi, get_bin, run_with_retry


# --- get_abi tests ---


class TestGetAbi:
    def test_returns_parsed_json(self, tmp_path):
        abi_data = [{"type": "function", "name": "transfer"}]
        abi_file = tmp_path / "Test.abi"
        abi_file.write_text(json.dumps(abi_data))

        with patch("opengradient.client._utils._ABI_DIR", tmp_path):
            result = get_abi("Test.abi")

        assert result == abi_data

    def test_raises_on_missing_file(self, tmp_path):
        with patch("opengradient.client._utils._ABI_DIR", tmp_path):
            with pytest.raises(FileNotFoundError):
                get_abi("NonExistent.abi")

    def test_raises_on_invalid_json(self, tmp_path):
        abi_file = tmp_path / "Bad.abi"
        abi_file.write_text("not valid json {{{")

        with patch("opengradient.client._utils._ABI_DIR", tmp_path):
            with pytest.raises(json.JSONDecodeError):
                get_abi("Bad.abi")


# --- get_bin tests ---


class TestGetBin:
    def test_returns_bytecode_with_prefix(self, tmp_path):
        bin_file = tmp_path / "Test.bin"
        bin_file.write_text("0x6060604052")

        with patch("opengradient.client._utils._BIN_DIR", tmp_path):
            result = get_bin("Test.bin")

        assert result == "0x6060604052"

    def test_adds_0x_prefix_if_missing(self, tmp_path):
        bin_file = tmp_path / "NoPre.bin"
        bin_file.write_text("6060604052")

        with patch("opengradient.client._utils._BIN_DIR", tmp_path):
            result = get_bin("NoPre.bin")

        assert result == "0x6060604052"

    def test_strips_whitespace(self, tmp_path):
        bin_file = tmp_path / "Spaced.bin"
        bin_file.write_text("  0x6060604052  \n")

        with patch("opengradient.client._utils._BIN_DIR", tmp_path):
            result = get_bin("Spaced.bin")

        assert result == "0x6060604052"

    def test_raises_on_missing_file(self, tmp_path):
        with patch("opengradient.client._utils._BIN_DIR", tmp_path):
            with pytest.raises(FileNotFoundError):
                get_bin("Missing.bin")


# --- run_with_retry tests ---


class TestRunWithRetry:
    def test_success_on_first_attempt(self):
        fn = MagicMock(return_value="ok")
        result = run_with_retry(fn, max_retries=3, retry_delay=0)

        assert result == "ok"
        assert fn.call_count == 1

    def test_raises_valueerror_for_zero_retries(self):
        with pytest.raises(ValueError, match="max_retries must be at least 1"):
            run_with_retry(lambda: None, max_retries=0)

    def test_raises_valueerror_for_negative_retries(self):
        with pytest.raises(ValueError, match="max_retries must be at least 1"):
            run_with_retry(lambda: None, max_retries=-1)

    def test_non_nonce_error_raises_immediately(self):
        fn = MagicMock(side_effect=RuntimeError("out of gas"))
        with pytest.raises(RuntimeError, match="out of gas"):
            run_with_retry(fn, max_retries=3, retry_delay=0)

        assert fn.call_count == 1

    def test_retries_on_nonce_too_low(self):
        fn = MagicMock(side_effect=[Exception("nonce too low"), "ok"])
        result = run_with_retry(fn, max_retries=3, retry_delay=0)

        assert result == "ok"
        assert fn.call_count == 2

    def test_retries_on_nonce_too_high(self):
        fn = MagicMock(side_effect=[Exception("nonce too high"), "ok"])
        result = run_with_retry(fn, max_retries=3, retry_delay=0)

        assert result == "ok"
        assert fn.call_count == 2

    def test_retries_on_invalid_nonce(self):
        fn = MagicMock(side_effect=[Exception("invalid nonce"), "ok"])
        result = run_with_retry(fn, max_retries=3, retry_delay=0)

        assert result == "ok"
        assert fn.call_count == 2

    def test_exhausts_retries_on_persistent_nonce_error(self):
        fn = MagicMock(side_effect=Exception("nonce too low"))
        with pytest.raises(RuntimeError, match="Transaction failed after 3 attempts"):
            run_with_retry(fn, max_retries=3, retry_delay=0)

        assert fn.call_count == 3

    def test_defaults_max_retries_when_none(self):
        fn = MagicMock(return_value="ok")
        result = run_with_retry(fn, max_retries=None, retry_delay=0)

        assert result == "ok"

    def test_nonce_error_case_insensitive(self):
        fn = MagicMock(side_effect=[Exception("NONCE TOO LOW"), "ok"])
        result = run_with_retry(fn, max_retries=3, retry_delay=0)

        assert result == "ok"
        assert fn.call_count == 2
