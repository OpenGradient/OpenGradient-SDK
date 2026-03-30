from unittest.mock import patch
import json

import numpy as np
import pytest

import opengradient.client._conversions as utils
import opengradient.types as types


@pytest.fixture
def sample_array_data():
    """Fixture providing sample array data for testing"""
    return [
        # Number tensors
        [
            ["tensor1", [[1000, 2], [2000, 2]], [2]],  # 10.00, 20.00 with 2 decimal places
            ["tensor2", [[500, 1]], [1]],  # 50.0 with 1 decimal place
        ],
        # String tensors
        [["str_tensor1", ["hello", "world"], [2]], ["str_tensor2", ["test"], [1]]],
        # JSON tensors
        [["json1", '{"key": "value"}'], ["json2", '{"numbers": [1, 2, 3]}']],
        # Is simulation result
        True,
    ]


def test_convert_array_basic(sample_array_data):
    """Test basic conversion with valid input data"""
    result = utils.convert_array_to_model_output(sample_array_data)

    # Test that result is a ModelOutput instance
    assert isinstance(result, types.ModelOutput)

    # Test number tensors
    assert len(result.numbers) == 2
    assert "tensor1" in result.numbers
    assert "tensor2" in result.numbers

    # Verify converted values
    np.testing.assert_array_almost_equal(result.numbers["tensor1"], np.array([10.00, 20.00]))
    np.testing.assert_array_almost_equal(result.numbers["tensor2"], np.array([50.0]))

    # Test string tensors
    assert len(result.strings) == 2
    assert "str_tensor1" in result.strings
    np.testing.assert_array_equal(result.strings["str_tensor1"], np.array(["hello", "world"]))

    # Test JSON tensors
    assert len(result.jsons) == 2
    assert "json1" in result.jsons
    assert result.jsons["json1"].item()["key"] == "value"
    assert result.jsons["json2"].item()["numbers"] == [1, 2, 3]

    # Test simulation result flag
    assert result.is_simulation_result is True


def test_convert_array_empty():
    """Test conversion with empty arrays"""
    empty_data = [[], [], [], False]
    result = utils.convert_array_to_model_output(empty_data)

    assert isinstance(result, types.ModelOutput)
    assert len(result.numbers) == 0
    assert len(result.strings) == 0
    assert len(result.jsons) == 0
    assert result.is_simulation_result is False


def test_convert_array_invalid_json():
    """Test that invalid JSON tensor data is skipped with a warning instead of crashing.

    Previously json.loads() was unguarded and would crash the entire conversion.
    After the fix, malformed JSON tensors are skipped and logged as warnings so
    all other valid tensors in the same response are still returned.
    """
    invalid_json_data = [
        [],  # Empty number tensors
        [],  # Empty string tensors
        [["invalid_json", '{"key": invalid}']],  # Invalid JSON
        False,
    ]

    import logging
    with pytest.warns(None) as warning_list:
        # Capture logging.warning calls via caplog-style check
        with patch("opengradient.client._conversions.logging") as mock_log:
            result = utils.convert_array_to_model_output(invalid_json_data)

    # Should not raise; result should have empty jsons dict (bad tensor skipped)
    assert isinstance(result, types.ModelOutput)
    assert result.jsons == {}
    # The warning should have been logged
    mock_log.warning.assert_called_once()
    assert "invalid_json" in mock_log.warning.call_args[0][0]


def test_convert_array_invalid_shape():
    """Test handling of invalid shape in tensors"""
    invalid_shape_data = [
        [["tensor1", [[1000, 2]], [3]]],  # Shape doesn't match data
        [],
        [],
        False,
    ]

    with pytest.raises(ValueError):
        utils.convert_array_to_model_output(invalid_shape_data)


def test_convert_array_large_numbers():
    """Test conversion of large numbers"""
    large_number_data = [
        [["large_tensor", [[1000000000, 8]], [1]]],  # 10.00000000
        [],
        [],
        False,
    ]

    result = utils.convert_array_to_model_output(large_number_data)
    np.testing.assert_array_almost_equal(result.numbers["large_tensor"], np.array([10.0]))


def test_convert_array_negative_numbers():
    """Test conversion of negative numbers"""
    negative_number_data = [
        [["neg_tensor", [[-1000, 2]], [1]]],  # -10.00
        [],
        [],
        False,
    ]

    result = utils.convert_array_to_model_output(negative_number_data)
    np.testing.assert_array_almost_equal(result.numbers["neg_tensor"], np.array([-10.0]))
