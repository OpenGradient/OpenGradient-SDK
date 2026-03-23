"""
Basic LangChain agent using OpenGradient.

Creates a simple ReAct agent with a tool, powered by an OpenGradient LLM.

Usage:
    export OG_PRIVATE_KEY="your_private_key"
    python examples/langchain_react_agent.py
"""

import os

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

import opengradient as og

private_key = os.environ["OG_PRIVATE_KEY"]

# One-time Permit2 approval for OPG spending (idempotent)
llm_client = og.LLM(private_key=private_key)
llm_client.ensure_opg_approval(opg_amount=5)

# Create the OpenGradient LangChain adapter
llm = og.agents.langchain_adapter(
    private_key=private_key,
    model_cid=og.TEE_LLM.GPT_4_1_2025_04_14,
    max_tokens=300,
    x402_settlement_mode=og.x402SettlementMode.INDIVIDUAL_FULL,
)


# Define a simple tool
@tool
def get_balance(account: str) -> str:
    """Returns the balance for a given account name."""
    balances = {"main": "1.25 ETH", "savings": "10.0 ETH", "treasury": "100.0 ETH"}
    return balances.get(account, "Account not found")


# Create a ReAct agent with the tool
agent = create_react_agent(llm, [get_balance])

# Run the agent
result = agent.invoke({"messages": [("user", "What is the balance of my 'treasury' account?")]})

print(result["messages"][-1].content)
