def has_azure_parameters(**kwargs) -> bool:
    azure_params = ["azure_base_url", "azure_deployment_name"]
    return any(kwargs.get(param) for param in azure_params)
