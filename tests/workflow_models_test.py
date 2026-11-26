"""Tests for workflow_models error handling and key validation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from opengradient.workflow_models.utils import read_workflow_wrapper
from opengradient.workflow_models.workflow_models import _extract_number


class TestExtractNumber:
    """Tests for _extract_number helper."""

    def test_extracts_valid_key(self):
        result = MagicMock()
        result.numbers = {"Y": np.float32(0.05)}
        assert _extract_number(result, "Y") == float(np.float32(0.05))

    def test_raises_on_missing_key(self):
        result = MagicMock()
        result.numbers = {"X": np.float32(0.05)}
        with pytest.raises(KeyError, match="Expected key 'Y' not found"):
            _extract_number(result, "Y")

    def test_raises_on_missing_numbers_attr(self):
        result = MagicMock(spec=[])  # no attributes
        with pytest.raises(KeyError, match="Expected key 'Y' not found"):
            _extract_number(result, "Y")

    def test_reports_available_keys(self):
        result = MagicMock()
        result.numbers = {"A": np.float32(1.0), "B": np.float32(2.0)}
        with pytest.raises(KeyError, match="Available keys:"):
            _extract_number(result, "missing")


class TestReadWorkflowWrapper:
    """Tests for read_workflow_wrapper error handling."""

    def test_keyerror_preserved_not_converted_to_runtime(self):
        """KeyError from missing output key should propagate as KeyError, not RuntimeError."""
        mock_alpha = MagicMock()
        mock_alpha.read_workflow_result.return_value = MagicMock()

        def bad_format(result):
            raise KeyError("regression_output")

        with pytest.raises(KeyError, match="regression_output"):
            read_workflow_wrapper(mock_alpha, "0xabc", bad_format)

    def test_runtime_error_chains_original(self):
        """Non-KeyError exceptions should be wrapped in RuntimeError with __cause__ set."""
        mock_alpha = MagicMock()
        mock_alpha.read_workflow_result.side_effect = ConnectionError("network down")

        with pytest.raises(RuntimeError, match="Error reading from workflow") as exc_info:
            read_workflow_wrapper(mock_alpha, "0xabc", lambda x: str(x))

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ConnectionError)

    def test_success_returns_workflow_output(self):
        """Successful read returns a WorkflowModelOutput."""
        mock_alpha = MagicMock()
        mock_alpha.read_workflow_result.return_value = "mock_result"

        output = read_workflow_wrapper(mock_alpha, "0xabc", lambda x: "formatted")
        assert output.result == "formatted"
        assert "0xabc" in output.block_explorer_link
