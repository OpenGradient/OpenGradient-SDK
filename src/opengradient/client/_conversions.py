# type: ignore

import json
import logging
from decimal import Decimal
from typing import Dict, List, Tuple

import numpy as np
from web3.datastructures import AttributeDict

from ..types import ModelOutput


def convert_to_fixed_point(number: float) -> Tuple[int, int]:
    """
    Converts input number to the Number tensor used by the sequencer.
    Also ensures that decimal is returned as a positive number.

    Returns a tuple of (value, decimal)
    """
    # Converting number to string in case this is a numpy float
    decimal_val = Decimal(str(number)).normalize()
    sign, digits, exponent = decimal_val.as_tuple()
    value = int("".join(map(str, digits)))
    if sign:
        value = -value

    if exponent >= 0:
        value *= 10**exponent
        decimals = 0
    else:
        decimals = -exponent

    logging.debug(f"Converted number {number} to fixed point value={value} decimals={decimals}")
    logging.debug(f"Types value={type(value)} decimals={type(decimals)}")
    return value, decimals


def convert_to_float32(value: int, decimals: int) -> np.float32:
    """
    Converts fixed point back into floating point

    Returns an np.float32 type
    """
    return np.float32(Decimal(value) / (10 ** Decimal(decimals)))


def convert_to_model_input(inputs: Dict[str, np.ndarray]) -> Tuple[List[Tuple[str, List[Tuple[int, int]]]], List[Tuple[str, List[str]]]]:
    """
    Expect SDK input to be a dict with the format
        key: tensor name
        value: np.array / list

    Return a tuple of (number tensors, string tensors) depending on the input type.
    Each number and string tensor converted to a numpy array and flattened and the shape saved.
    """
    logging.debug("Converting the following input dictionary to ModelInput: %s", inputs)
    number_tensors = []
    string_tensors = []
    json_tensors = []
    for tensor_name, tensor_data in inputs.items():
        # Convert to NP array if list or single object
        if isinstance(tensor_data, list):
            logging.debug(f"\tConverting {tensor_data} to np array")
            tensor_data = np.array(tensor_data)

        if isinstance(tensor_data, (str, int, float)):
            logging.debug(f"\tConverting single entry {tensor_data} to a list")
            tensor_data = np.array([tensor_data])

        # Check if type is np array
        if not isinstance(tensor_data, np.ndarray):
            raise TypeError("Inference input must be list, numpy array, or type (str, int, float): %s" % type(tensor_data))

        # Flatten list and retain shape
        shape = tensor_data.shape
        flat_data = tensor_data.flatten()
        logging.debug("Shape and flattened data: %s, %s", shape, flat_data)

        # Parse into number and string tensors
        if issubclass(tensor_data.dtype.type, np.floating):
            # Convert to fixed-point tuples
            data_type = np.dtype([("value", int), ("decimal", int)])
            converted_tensor_data = np.array([convert_to_fixed_point(i) for i in flat_data], dtype=data_type)

            input = (tensor_name, converted_tensor_data.tolist(), shape)
            logging.debug("\tFloating tensor input: %s", input)

            number_tensors.append(input)
        elif issubclass(tensor_data.dtype.type, np.integer):
            # Convert to fixed-point tuples
            data_type = np.dtype([("value", int), ("decimal", int)])
            converted_tensor_data = np.array([convert_to_fixed_point(int(i)) for i in flat_data], dtype=data_type)

            input = (tensor_name, converted_tensor_data.tolist(), shape)
            logging.debug("\tInteger tensor input: %s", input)

            number_tensors.append(input)
        elif issubclass(tensor_data.dtype.type, np.str_):
            # TODO (Kyle): Add shape into here as well
            input = (tensor_name, [s for s in flat_data])
            logging.debug("\tString tensor input: %s", input)

            string_tensors.append(input)
        else:
            raise TypeError(f"Data type {tensor_data.dtype.type} not recognized")

    logging.debug("Number tensors: %s", number_tensors)
    logging.debug("Number tensor types: %s", [type(item) for item in number_tensors])
    logging.debug("String tensors: %s", string_tensors)
    logging.debug("String tensor types: %s", [type(item) for item in string_tensors])
    return number_tensors, string_tensors


def convert_to_model_output(event_data: AttributeDict) -> Dict[str, np.ndarray]:
    """
    Converts inference output into a user-readable output.
    Expects the inference node to return a dict with the format:
        key: output_name (str)
        value: (output_array (list), shape (list)) (tuple)

    We need to reshape each output array using the shape parameter in order to get the array
    back into its original shape.
    """
    output_dict = {}
    output = event_data.get("output", {})

    if isinstance(output, (AttributeDict, dict)):
        for tensor in output.get("numbers", []):
            if isinstance(tensor, (AttributeDict, dict)):
                name = tensor.get("name")
                shape = tensor.get("shape")
                values = []
                # Convert from fixed point back into np.float32
                for v in tensor.get("values", []):
                    if isinstance(v, (AttributeDict, dict)):
                        values.append(convert_to_float32(value=int(v.get("value")), decimals=int(v.get("decimals"))))
                    else:
                        logging.warning(f"Unexpected number type: {type(v)}")
                output_dict[name] = np.array(values).reshape(shape)
            else:
                logging.warning(f"Unexpected tensor type: {type(tensor)}")

        # Parse strings
        for tensor in output.get("strings", []):
            if isinstance(tensor, (AttributeDict, dict)):
                name = tensor.get("name")
                shape = tensor.get("shape")
                values = tensor.get("values", [])
                output_dict[name] = np.array(values).reshape(shape)
            else:
                logging.warning(f"Unexpected tensor type: {type(tensor)}")

        # Parse JSON dicts
        for tensor in output.get("jsons", []):
            if isinstance(tensor, (AttributeDict, dict)):
                name = tensor.get("name")
                value = tensor.get("value")
                if name is None:
                    logging.warning("JSON tensor is missing 'name' field; skipping.")
                    continue
                try:
                    output_dict[name] = np.array(json.loads(value))
                except (json.JSONDecodeError, TypeError) as e:
                    logging.warning(
                        f"Failed to parse JSON tensor '{name}' — value={value!r}: {e}. Skipping."
                    )
            else:
                logging.warning(f"Unexpected tensor type: {type(tensor)}")

    else:
        logging.warning(f"Unexpected output type: {type(output)}")

    logging.debug(f"Parsed output: {output_dict}")
    return output_dict


def convert_array_to_model_output(array_data: List) -> ModelOutput:
    """
    Converts inference output (in array form) into a user-readable ModelOutput class.
    This expects data from the smart contract returned as 4 element array:
        array_data[0] = NumberTensor
        array_data[1] = StringTensor
        array_data[2] = JsonTensor
        array_data[3] = Bool
    """
    # Parse number tensors
    number_data = {}
    for tensor in array_data[0]:
        name = tensor[0]
        values = tensor[1]
        shape = tensor[2]

        # Convert from fixed point into np.float32
        converted_values = []
        for value in values:
            converted_values.append(convert_to_float32(value=value[0], decimals=value[1]))

        number_data[name] = np.array(converted_values).reshape(shape)

    # Parse string tensors
    string_data = {}
    for tensor in array_data[1]:
        name = tensor[0]
        values = tensor[1]
        shape = tensor[2]
        string_data[name] = np.array(values).reshape(shape)

    # Parse JSON tensors
    json_data = {}
    for tensor in array_data[2]:
        name = tensor[0]
        value = tensor[1]
        if name is None:
            logging.warning("JSON tensor entry is missing a name; skipping.")
            continue
        try:
            json_data[name] = np.array(json.loads(value))
        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(
                f"Failed to parse JSON tensor '{name}' — value={value!r}: {e}. Skipping."
            )

    return ModelOutput(
        numbers=number_data,
        strings=string_data,
        jsons=json_data,
        is_simulation_result=array_data[3],
    )
