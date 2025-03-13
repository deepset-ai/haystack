# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install azure-identity") as azure_import:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider


def default_azure_ad_token_provider() -> str:
    """
    Get a Azure AD token using the DefaultAzureCredential and the "https://cognitiveservices.azure.com/.default" scope.
    """
    azure_import.check()
    return get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")()
