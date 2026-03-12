"""
Alpha Testnet features for OpenGradient SDK.

This module contains features that are only available on the Alpha Testnet,
including on-chain ONNX model inference, workflow management, and ML model execution.
"""

import base64
import json
import urllib.parse
from typing import Dict, List, Optional, Union

import numpy as np
import requests
from eth_account.account import LocalAccount
from web3 import Web3
from web3.exceptions import ContractLogicError
from web3.logs import DISCARD

from ..defaults import DEFAULT_SCHEDULER_ADDRESS
from ..types import HistoricalInputQuery, InferenceMode, InferenceResult, ModelOutput, SchedulerParams
from ._conversions import convert_array_to_model_output, convert_to_model_input, convert_to_model_output
from ._utils import get_abi, get_bin, run_with_retry

# How much time we wait for txn to be included in chain
INFERENCE_TX_TIMEOUT = 120
REGULAR_TX_TIMEOUT = 30

PRECOMPILE_CONTRACT_ADDRESS = "0x00000000000000000000000000000000000000F4"


class Alpha:
    """
    Alpha Testnet features namespace.

    This class provides access to features that are only available on the Alpha Testnet,
    including on-chain ONNX model inference, workflow deployment, and execution.

    Usage:
        client = og.Client(...)
        result = client.alpha.infer(model_cid, InferenceMode.VANILLA, model_input)
        result = client.alpha.new_workflow(model_cid, input_query, input_tensor_name)
    """

    def __init__(
        self,
        blockchain: Web3,
        wallet_account: LocalAccount,
        inference_hub_contract_address: str,
        api_url: str,
    ):
        self._blockchain = blockchain
        self._wallet_account = wallet_account
        self._inference_hub_contract_address = inference_hub_contract_address
        self._api_url = api_url
        self._inference_abi = None
        self._precompile_abi = None

    @property
    def inference_abi(self) -> dict:
        if self._inference_abi is None:
            self._inference_abi = get_abi("inference.abi")
        return self._inference_abi

    @property
    def precompile_abi(self) -> dict:
        if self._precompile_abi is None:
            self._precompile_abi = get_abi("InferencePrecompile.abi")
        return self._precompile_abi

    def infer(
        self,
        model_cid: str,
        inference_mode: InferenceMode,
        model_input: Dict[str, Union[str, int, float, List, np.ndarray]],
        max_retries: Optional[int] = None,
    ) -> InferenceResult:
        """
        Perform inference on a model.

        Args:
            model_cid (str): The unique content identifier for the model from IPFS.
            inference_mode (InferenceMode): The inference mode.
            model_input (Dict[str, Union[str, int, float, List, np.ndarray]]): The input data for the model.
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 5.

        Returns:
            InferenceResult (InferenceResult): A dataclass object containing the transaction hash and model output.
                transaction_hash (str): Blockchain hash for the transaction
                model_output (Dict[str, np.ndarray]): Output of the ONNX model

        Raises:
            RuntimeError: If the inference fails.
        """

        def execute_transaction():
            contract = self._blockchain.eth.contract(address=self._inference_hub_contract_address, abi=self.inference_abi)
            precompile_contract = self._blockchain.eth.contract(address=PRECOMPILE_CONTRACT_ADDRESS, abi=self.precompile_abi)

            inference_mode_uint8 = inference_mode.value
            converted_model_input = convert_to_model_input(model_input)

            run_function = contract.functions.run(model_cid, inference_mode_uint8, converted_model_input)

            tx_hash, tx_receipt = self._send_tx_with_revert_handling(run_function)
            parsed_logs = contract.events.InferenceResult().process_receipt(tx_receipt, errors=DISCARD)
            if len(parsed_logs) < 1:
                raise RuntimeError("InferenceResult event not found in transaction logs")

            # TODO: This should return a ModelOutput class object
            model_output = convert_to_model_output(parsed_logs[0]["args"])
            if len(model_output) == 0:
                # check inference directly from node
                parsed_logs = precompile_contract.events.ModelInferenceEvent().process_receipt(tx_receipt, errors=DISCARD)
                inference_id = parsed_logs[0]["args"]["inferenceID"]
                inference_result = self._get_inference_result_from_node(inference_id, inference_mode)
                model_output = convert_to_model_output(inference_result)

            return InferenceResult(tx_hash.hex(), model_output)

        return run_with_retry(execute_transaction, max_retries)

    def _send_tx_with_revert_handling(self, run_function):
        """
        Execute a blockchain transaction with revert error.

        Args:
            run_function: Function that executes the transaction

        Returns:
            tx_hash: Transaction hash
            tx_receipt: Transaction receipt

        Raises:
            Exception: If transaction fails or gas estimation fails
        """
        nonce = self._blockchain.eth.get_transaction_count(self._wallet_account.address, "pending")
        try:
            estimated_gas = run_function.estimate_gas({"from": self._wallet_account.address})
        except ContractLogicError as e:
            try:
                run_function.call({"from": self._wallet_account.address})

            except ContractLogicError as call_err:
                raise ContractLogicError(f"simulation failed with revert reason: {call_err.args[0]}")

            raise ContractLogicError(f"simulation failed with no revert reason. Reason: {e}")

        gas_limit = int(estimated_gas * 3)

        transaction = run_function.build_transaction(
            {
                "from": self._wallet_account.address,
                "nonce": nonce,
                "gas": gas_limit,
                "gasPrice": self._blockchain.eth.gas_price,
            }
        )

        signed_tx = self._wallet_account.sign_transaction(transaction)
        tx_hash = self._blockchain.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_receipt = self._blockchain.eth.wait_for_transaction_receipt(tx_hash, timeout=INFERENCE_TX_TIMEOUT)

        if tx_receipt["status"] == 0:
            try:
                run_function.call({"from": self._wallet_account.address})

            except ContractLogicError as call_err:
                raise ContractLogicError(f"Transaction failed with revert reason: {call_err.args[0]}")

            raise ContractLogicError(f"Transaction failed with no revert reason. Receipt: {tx_receipt}")

        return tx_hash, tx_receipt

    def _get_inference_result_from_node(self, inference_id: str, inference_mode: InferenceMode) -> Dict:
        """
        Get the inference result from node.

        Args:
            inference_id (str): Inference id for a inference request

        Returns:
            Dict: The inference result as returned by the node

        Raises:
            RuntimeError: If the request fails or returns an error
        """
        try:
            encoded_id = urllib.parse.quote(inference_id, safe="")
            url = f"{self._api_url}/artela-network/artela-rollkit/inference/tx/{encoded_id}"

            response = requests.get(url)
            if response.status_code == 200:
                resp = response.json()
                inference_result = resp.get("inference_results", {})
                if inference_result:
                    decoded_bytes = base64.b64decode(inference_result[0])
                    decoded_string = decoded_bytes.decode("utf-8")
                    output = json.loads(decoded_string).get("InferenceResult", {})
                    if output is None:
                        raise RuntimeError("Missing InferenceResult in inference output")

                    match inference_mode:
                        case InferenceMode.VANILLA:
                            if "VanillaResult" not in output:
                                raise RuntimeError("Missing VanillaResult in inference output")
                            if "model_output" not in output["VanillaResult"]:
                                raise RuntimeError("Missing model_output in VanillaResult")
                            return {"output": output["VanillaResult"]["model_output"]}

                        case InferenceMode.TEE:
                            if "TeeNodeResult" not in output:
                                raise RuntimeError("Missing TeeNodeResult in inference output")
                            if "Response" not in output["TeeNodeResult"]:
                                raise RuntimeError("Missing Response in TeeNodeResult")
                            if "VanillaResponse" in output["TeeNodeResult"]["Response"]:
                                if "model_output" not in output["TeeNodeResult"]["Response"]["VanillaResponse"]:
                                    raise RuntimeError("Missing model_output in VanillaResponse")
                                return {"output": output["TeeNodeResult"]["Response"]["VanillaResponse"]["model_output"]}

                            else:
                                raise RuntimeError("Missing VanillaResponse in TeeNodeResult Response")

                        case InferenceMode.ZKML:
                            if "ZkmlResult" not in output:
                                raise RuntimeError("Missing ZkmlResult in inference output")
                            if "model_output" not in output["ZkmlResult"]:
                                raise RuntimeError("Missing model_output in ZkmlResult")
                            return {"output": output["ZkmlResult"]["model_output"]}

                        case _:
                            raise ValueError(f"Invalid inference mode: {inference_mode}")
                else:
                    return None

            else:
                raise RuntimeError(f"Failed to get inference result: HTTP {response.status_code}")

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to get inference result: {str(e)}")
        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to get inference result: {str(e)}")

    def new_workflow(
        self,
        model_cid: str,
        input_query: HistoricalInputQuery,
        input_tensor_name: str,
        scheduler_params: Optional[SchedulerParams] = None,
    ) -> str:
        """
        Deploy a new workflow contract with the specified parameters.

        This function deploys a new workflow contract on OpenGradient that connects
        an AI model with its required input data. When executed, the workflow will fetch
        the specified model, evaluate the input query to get data, and perform inference.

        The workflow can be set to execute manually or automatically via a scheduler.

        Args:
            model_cid (str): CID of the model to be executed from the Model Hub
            input_query (HistoricalInputQuery): Input definition for the model inference,
                will be evaluated at runtime for each inference
            input_tensor_name (str): Name of the input tensor expected by the model
            scheduler_params (Optional[SchedulerParams]): Scheduler configuration for automated execution:
                - frequency: Execution frequency in seconds
                - duration_hours: How long the schedule should live for

        Returns:
            str: Deployed contract address. If scheduler_params was provided, the workflow
                 will be automatically executed according to the specified schedule.

        Raises:
            Exception: If transaction fails or gas estimation fails
        """
        # Get contract ABI and bytecode
        abi = get_abi("PriceHistoryInference.abi")
        bytecode = get_bin("PriceHistoryInference.bin")

        def deploy_transaction():
            contract = self._blockchain.eth.contract(abi=abi, bytecode=bytecode)
            query_tuple = input_query.to_abi_format()
            constructor_args = [model_cid, input_tensor_name, query_tuple]

            try:
                # Estimate gas needed
                estimated_gas = contract.constructor(*constructor_args).estimate_gas({"from": self._wallet_account.address})
                gas_limit = int(estimated_gas * 1.2)
            except Exception as e:
                print(f"Gas estimation failed: {str(e)}")
                gas_limit = 5000000  # Conservative fallback
                print(f"Using fallback gas limit: {gas_limit}")

            transaction = contract.constructor(*constructor_args).build_transaction(
                {
                    "from": self._wallet_account.address,
                    "nonce": self._blockchain.eth.get_transaction_count(self._wallet_account.address, "pending"),
                    "gas": gas_limit,
                    "gasPrice": self._blockchain.eth.gas_price,
                    "chainId": self._blockchain.eth.chain_id,
                }
            )

            signed_txn = self._wallet_account.sign_transaction(transaction)
            tx_hash = self._blockchain.eth.send_raw_transaction(signed_txn.raw_transaction)

            tx_receipt = self._blockchain.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

            if tx_receipt["status"] == 0:
                raise Exception(f"Contract deployment failed, transaction hash: {tx_hash.hex()}")

            return tx_receipt.contractAddress

        contract_address = run_with_retry(deploy_transaction)

        if scheduler_params:
            self._register_with_scheduler(contract_address, scheduler_params)

        return contract_address

    def _register_with_scheduler(self, contract_address: str, scheduler_params: SchedulerParams) -> None:
        """
        Register the deployed workflow contract with the scheduler for automated execution.

        Args:
            contract_address (str): Address of the deployed workflow contract
            scheduler_params (SchedulerParams): Scheduler configuration containing:
                - frequency: Execution frequency in seconds
                - duration_hours: How long to run in hours
                - end_time: Unix timestamp when scheduling should end

        Raises:
            Exception: If registration with scheduler fails. The workflow contract will
                      still be deployed and can be executed manually.
        """
        scheduler_abi = get_abi("WorkflowScheduler.abi")

        # Scheduler contract address
        scheduler_address = DEFAULT_SCHEDULER_ADDRESS
        scheduler_contract = self._blockchain.eth.contract(address=scheduler_address, abi=scheduler_abi)

        try:
            # Register the workflow with the scheduler
            scheduler_tx = scheduler_contract.functions.registerTask(
                contract_address, scheduler_params.end_time, scheduler_params.frequency
            ).build_transaction(
                {
                    "from": self._wallet_account.address,
                    "gas": 300000,
                    "gasPrice": self._blockchain.eth.gas_price,
                    "nonce": self._blockchain.eth.get_transaction_count(self._wallet_account.address, "pending"),
                    "chainId": self._blockchain.eth.chain_id,
                }
            )

            signed_scheduler_tx = self._wallet_account.sign_transaction(scheduler_tx)
            scheduler_tx_hash = self._blockchain.eth.send_raw_transaction(signed_scheduler_tx.raw_transaction)
            self._blockchain.eth.wait_for_transaction_receipt(scheduler_tx_hash, timeout=REGULAR_TX_TIMEOUT)
        except Exception as e:
            print(f"Error registering contract with scheduler: {str(e)}")
            print("  The workflow contract is still deployed and can be executed manually.")

    def read_workflow_result(self, contract_address: str) -> ModelOutput:
        """
        Reads the latest inference result from a deployed workflow contract.

        Args:
            contract_address (str): Address of the deployed workflow contract

        Returns:
            ModelOutput: The inference result from the contract

        Raises:
            ContractLogicError: If the transaction fails
            Web3Error: If there are issues with the web3 connection or contract interaction
        """
        # Get the contract interface
        contract = self._blockchain.eth.contract(
            address=Web3.to_checksum_address(contract_address), abi=get_abi("PriceHistoryInference.abi")
        )

        # Get the result
        result = contract.functions.getInferenceResult().call()

        return convert_array_to_model_output(result)

    def run_workflow(self, contract_address: str) -> ModelOutput:
        """
        Triggers the run() function on a deployed workflow contract and returns the result.

        Args:
            contract_address (str): Address of the deployed workflow contract

        Returns:
            ModelOutput: The inference result from the contract

        Raises:
            ContractLogicError: If the transaction fails
            Web3Error: If there are issues with the web3 connection or contract interaction
        """
        # Get the contract interface
        contract = self._blockchain.eth.contract(
            address=Web3.to_checksum_address(contract_address), abi=get_abi("PriceHistoryInference.abi")
        )

        # Call run() function
        nonce = self._blockchain.eth.get_transaction_count(self._wallet_account.address, "pending")

        run_function = contract.functions.run()
        transaction = run_function.build_transaction(
            {
                "from": self._wallet_account.address,
                "nonce": nonce,
                "gas": 30000000,
                "gasPrice": self._blockchain.eth.gas_price,
                "chainId": self._blockchain.eth.chain_id,
            }
        )

        signed_txn = self._wallet_account.sign_transaction(transaction)
        tx_hash = self._blockchain.eth.send_raw_transaction(signed_txn.raw_transaction)
        tx_receipt = self._blockchain.eth.wait_for_transaction_receipt(tx_hash, timeout=INFERENCE_TX_TIMEOUT)

        if tx_receipt.status == 0:
            raise ContractLogicError(f"Run transaction failed. Receipt: {tx_receipt}")

        # Get the inference result from the contract
        result = contract.functions.getInferenceResult().call()

        return convert_array_to_model_output(result)

    def read_workflow_history(self, contract_address: str, num_results: int) -> List[ModelOutput]:
        """
        Gets historical inference results from a workflow contract.

        Retrieves the specified number of most recent inference results from the contract's
        storage, with the most recent result first.

        Args:
            contract_address (str): Address of the deployed workflow contract
            num_results (int): Number of historical results to retrieve

        Returns:
            List[ModelOutput]: List of historical inference results
        """
        contract = self._blockchain.eth.contract(
            address=Web3.to_checksum_address(contract_address), abi=get_abi("PriceHistoryInference.abi")
        )

        results = contract.functions.getLastInferenceResults(num_results).call()
        return [convert_array_to_model_output(result) for result in results]
