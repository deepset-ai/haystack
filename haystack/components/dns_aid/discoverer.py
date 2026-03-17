"""DNS-AID Haystack pipeline components.

Provides Discoverer and Publisher components that slot into Haystack pipelines.
"""

from __future__ import annotations

from typing import Optional

from haystack import component, default_from_dict, default_to_dict


@component
class DnsAidDiscoverer:
    """Haystack component that discovers AI agents via DNS-AID.

    Queries DNS SVCB records to find published agents at a domain.

    Example::

        from haystack import Pipeline
        from haystack_dns_aid import DnsAidDiscoverer

        pipe = Pipeline()
        pipe.add_component("discoverer", DnsAidDiscoverer(domain="agents.example.com"))
        result = pipe.run({"discoverer": {}})
    """

    def __init__(
        self,
        domain: str = "",
        protocol: Optional[str] = None,
    ) -> None:
        self.domain = domain
        self.protocol = protocol

    @component.output_types(agents=str)
    def run(
        self,
        domain: Optional[str] = None,
        protocol: Optional[str] = None,
        name: Optional[str] = None,
    ) -> dict[str, str]:
        """Discover agents. Returns JSON string of discovered agents."""
        from dns_aid.integrations import DnsAidOperations

        ops = DnsAidOperations()
        result = ops.discover_sync(
            domain=domain or self.domain,
            protocol=protocol or self.protocol,
            name=name,
        )
        return {"agents": result}

    def to_dict(self) -> dict:
        return default_to_dict(self, domain=self.domain, protocol=self.protocol)

    @classmethod
    def from_dict(cls, data: dict) -> DnsAidDiscoverer:
        return default_from_dict(cls, data)


@component
class DnsAidPublisher:
    """Haystack component that publishes an AI agent to DNS via DNS-AID.

    Example::

        from haystack_dns_aid import DnsAidPublisher

        publisher = DnsAidPublisher(backend_name="route53")
        result = publisher.run(
            name="my-agent",
            domain="agents.example.com",
            endpoint="mcp.example.com",
        )
    """

    def __init__(self, backend_name: Optional[str] = None) -> None:
        self.backend_name = backend_name

    @component.output_types(result=str)
    def run(
        self,
        name: str,
        domain: str,
        protocol: str = "mcp",
        endpoint: str = "",
        port: int = 443,
    ) -> dict[str, str]:
        """Publish agent to DNS. Returns JSON result."""
        from dns_aid.integrations import DnsAidOperations

        ops = DnsAidOperations(backend_name=self.backend_name)
        result = ops.publish_sync(
            name=name, domain=domain, protocol=protocol, endpoint=endpoint, port=port
        )
        return {"result": result}

    def to_dict(self) -> dict:
        return default_to_dict(self, backend_name=self.backend_name)

    @classmethod
    def from_dict(cls, data: dict) -> DnsAidPublisher:
        return default_from_dict(cls, data)
