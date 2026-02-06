# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Nory x402 Payment Tools for Haystack.

Tools for AI agents to make payments using the x402 HTTP protocol.
Supports Solana and 7 EVM chains with sub-400ms settlement.
"""

from typing import Annotated

import requests

from haystack.tools.from_function import tool

NORY_API_BASE = "https://noryx402.com"


def _get_headers(api_key: str | None = None, with_json: bool = False) -> dict[str, str]:
    """Get request headers with optional auth."""
    headers: dict[str, str] = {}
    if with_json:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


@tool
def nory_get_payment_requirements(
    resource: Annotated[str, "The resource path requiring payment (e.g., /api/premium/data)"],
    amount: Annotated[str, "Amount in human-readable format (e.g., '0.10' for $0.10 USDC)"],
    network: Annotated[
        str | None,
        "Preferred blockchain network (solana-mainnet, base-mainnet, polygon-mainnet, etc.)",
    ] = None,
    api_key: Annotated[str | None, "Nory API key (optional for public endpoints)"] = None,
) -> str:
    """Get x402 payment requirements for accessing a paid resource.

    Use this when you encounter an HTTP 402 Payment Required response
    and need to know how much to pay and where to send payment.

    Returns payment requirements including amount, supported networks, and wallet address.
    """
    params = {"resource": resource, "amount": amount}
    if network:
        params["network"] = network

    response = requests.get(
        f"{NORY_API_BASE}/api/x402/requirements",
        params=params,
        headers=_get_headers(api_key),
        timeout=30,
    )
    response.raise_for_status()
    return response.text


@tool
def nory_verify_payment(
    payload: Annotated[str, "Base64-encoded payment payload containing signed transaction"],
    api_key: Annotated[str | None, "Nory API key (optional for public endpoints)"] = None,
) -> str:
    """Verify a signed payment transaction before settlement.

    Use this to validate that a payment transaction is correct
    before submitting it to the blockchain.

    Returns verification result including validity and payer info.
    """
    response = requests.post(
        f"{NORY_API_BASE}/api/x402/verify",
        json={"payload": payload},
        headers=_get_headers(api_key, with_json=True),
        timeout=30,
    )
    response.raise_for_status()
    return response.text


@tool
def nory_settle_payment(
    payload: Annotated[str, "Base64-encoded payment payload"],
    api_key: Annotated[str | None, "Nory API key (optional for public endpoints)"] = None,
) -> str:
    """Settle a payment on-chain.

    Use this to submit a verified payment transaction to the blockchain.
    Settlement typically completes in under 400ms.

    Returns settlement result including transaction ID.
    """
    response = requests.post(
        f"{NORY_API_BASE}/api/x402/settle",
        json={"payload": payload},
        headers=_get_headers(api_key, with_json=True),
        timeout=30,
    )
    response.raise_for_status()
    return response.text


@tool
def nory_lookup_transaction(
    transaction_id: Annotated[str, "Transaction ID or signature"],
    network: Annotated[str, "Network where the transaction was submitted"],
    api_key: Annotated[str | None, "Nory API key (optional for public endpoints)"] = None,
) -> str:
    """Look up transaction status.

    Use this to check the status of a previously submitted payment.

    Returns transaction details including status and confirmations.
    """
    response = requests.get(
        f"{NORY_API_BASE}/api/x402/transactions/{transaction_id}",
        params={"network": network},
        headers=_get_headers(api_key),
        timeout=30,
    )
    response.raise_for_status()
    return response.text


@tool
def nory_health_check() -> str:
    """Check Nory service health.

    Use this to verify the payment service is operational
    and see supported networks.

    Returns health status and supported networks.
    """
    response = requests.get(f"{NORY_API_BASE}/api/x402/health", timeout=30)
    response.raise_for_status()
    return response.text


# Convenience list of all Nory x402 tools
NORY_X402_TOOLS = [
    nory_get_payment_requirements,
    nory_verify_payment,
    nory_settle_payment,
    nory_lookup_transaction,
    nory_health_check,
]
