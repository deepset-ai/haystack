---
title: "OAuth"
id: integrations-oauth
description: "OAuth integration for Haystack"
slug: "/integrations-oauth"
---


## haystack_integrations.components.connectors.oauth.resolver

### OAuthTokenResolver

Resolves an OAuth access token at pipeline runtime and emits it on the `access_token` output socket.

The resolver component is a thin wrapper over a pluggable token source that decides *where* the token comes from:
a standalone OAuth refresh grant (`OAuthRefreshTokenSource`), a client-credentials grant
(`OAuthClientCredentialsSource`), a per-request token exchange (`OAuthTokenExchangeSource`), a Microsoft Entra
Agent ID acting on behalf of a user (`EntraAgentOBOTokenSource`), a static long-lived token
(`OAuthStaticTokenSource`), or a custom source you provide. A downstream component (for
example a SharePoint or Google Drive retriever) consumes the token via a normal connection and never knows how
it was resolved.

The run input depends on the token source. A source that needs a per-request credential (it sets
`requires_subject_token = True`, like `OAuthTokenExchangeSource`) makes the resolver declare a **mandatory**
`subject_token` input — a controller-injected per-request credential (for example an incoming user assertion),
not chosen by an end user, and accepted as either a `Secret` or a plain `str`. A config-only source declares no
run input, so the resolver is a source node.

### Usage example

```python
from haystack.utils import Secret
from haystack_integrations.components.connectors.oauth import OAuthTokenResolver
from haystack_integrations.utils.oauth import OAuthRefreshTokenSource

resolver = OAuthTokenResolver(
    token_source=OAuthRefreshTokenSource(
        token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
        client_id="aaa-bbb-ccc",
        refresh_token=Secret.from_env_var("MS_REFRESH_TOKEN"),
        scopes=["https://graph.microsoft.com/Files.Read.All", "offline_access"],
    ),
)
access_token = resolver.run()["access_token"]
```

#### __init__

```python
__init__(token_source: TokenSource | SubjectTokenSource) -> None
```

Initialize the resolver.

**Parameters:**

- **token_source** (<code>TokenSource | SubjectTokenSource</code>) – The strategy that resolves the access token. If it sets `requires_subject_token = True`
  (for example `OAuthTokenExchangeSource`), the resolver declares a mandatory `subject_token` run input;
  otherwise the resolver takes no run input.

**Raises:**

- <code>OAuthConfigError</code> – If `token_source` does not implement a token-source protocol.

#### run

```python
run(**kwargs: Any) -> dict[str, Secret]
```

Resolve an access token and emit it.

**Parameters:**

- **kwargs** (<code>Any</code>) – Carries `subject_token` when the configured source requires it (declared as a mandatory
  input in that case, injected by the application/controller per request as a `Secret` or a plain `str`).
  For config-only sources no input is declared and `kwargs` is empty.

**Returns:**

- <code>dict\[str, Secret\]</code> – A dictionary with a single `access_token` key holding the bearer token as a `Secret`. Call
  `resolve_value()` on it to read the token string.

**Raises:**

- <code>OAuthConfigError</code> – If the source requires a `subject_token` but it is missing, empty, or a `Secret`
  that does not resolve.
- <code>TokenRefreshError</code> – If the source returns an empty access token.

#### run_async

```python
run_async(**kwargs: Any) -> dict[str, Secret]
```

Asynchronously resolve an access token and emit it.

**Parameters:**

- **kwargs** (<code>Any</code>) – Carries `subject_token` when the configured source requires it, as a `Secret` or a plain
  `str`.

**Returns:**

- <code>dict\[str, Secret\]</code> – A dictionary with a single `access_token` key holding the bearer token as a `Secret`. Call
  `resolve_value()` on it to read the token string.

**Raises:**

- <code>OAuthConfigError</code> – If the source requires a `subject_token` but it is missing, empty, or a `Secret`
  that does not resolve.
- <code>TokenRefreshError</code> – If the source returns an empty access token.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized component as a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OAuthTokenResolver
```

Deserialize this component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary representation of this component.

**Returns:**

- <code>OAuthTokenResolver</code> – The deserialized component instance.

**Raises:**

- <code>ImportError</code> – If the serialized `token_source` type cannot be imported.

## haystack_integrations.utils.oauth.entra

### EntraAgentOBOTokenSource

Resolves access tokens for a Microsoft Entra Agent ID acting on behalf of the signed-in user.

It implements the
[agent on-behalf-of flow](https://learn.microsoft.com/en-us/entra/agent-id/agent-on-behalf-of-oauth-flow)
for agent identities created from an agent identity blueprint. Each resolution takes the user's token as the
per-request `subject_token` and performs up to two requests against the tenant's token endpoint:

1. The blueprint authenticates with its own credential and obtains an intermediate token for the agent identity.
   This token is cached until shortly before it expires.
1. The agent identity exchanges that token and the user's token for an access token to the requested resource.
   The result is cached per user token.

Configure exactly one credential for the blueprint:

- `client_secret`: simplest; Microsoft advises against client secrets in production.
- `certificate` and `private_key`: requires `pip install "oauth-haystack[jwt]"`.
- `client_assertion_source`: any other config-only token source that yields a client assertion for the blueprint,
  for example the token of a managed identity that is registered as a federated identity credential of the
  blueprint.

What Microsoft Entra expects, and the errors you get otherwise:

- The user's token must be issued for the blueprint (its `aud` claim is the blueprint's client ID). A token for
  another resource, such as Microsoft Graph, is rejected with `AADSTS50013`.
- The agent identity needs the requested delegated permissions. They are granted to it by an administrator,
  either directly or as inheritable permissions on the blueprint, because an agent identity cannot ask the user
  for consent itself. Without them the exchange fails with `AADSTS82014`.

An access token is valid for a single resource, so use one resolver per downstream service.

### Usage example

```python
from haystack.utils import Secret
from haystack_integrations.components.connectors.oauth import OAuthTokenResolver
from haystack_integrations.utils.oauth import EntraAgentOBOTokenSource

resolver = OAuthTokenResolver(
    token_source=EntraAgentOBOTokenSource(
        tenant_id="<tenant-id>",
        blueprint_client_id="<blueprint-client-id>",
        agent_identity_client_id="<agent-identity-client-id>",
        scopes=["https://graph.microsoft.com/.default"],
        client_secret=Secret.from_env_var("ENTRA_BLUEPRINT_CLIENT_SECRET"),
    ),
)
# `user_token` is the signed-in user's access token for the blueprint, injected per request.
access_token = resolver.run(subject_token=user_token)["access_token"]
```

The same source authenticating the blueprint with a certificate instead of a client secret:

```python
token_source = EntraAgentOBOTokenSource(
    tenant_id="<tenant-id>",
    blueprint_client_id="<blueprint-client-id>",
    agent_identity_client_id="<agent-identity-client-id>",
    scopes=["https://graph.microsoft.com/.default"],
    certificate=Secret.from_env_var("ENTRA_BLUEPRINT_CERTIFICATE"),
    private_key=Secret.from_env_var("ENTRA_BLUEPRINT_PRIVATE_KEY"),
)
```

#### __init__

```python
__init__(
    tenant_id: str,
    blueprint_client_id: str,
    agent_identity_client_id: str,
    scopes: list[str],
    *,
    client_secret: Secret | None = None,
    certificate: Secret | None = None,
    private_key: Secret | None = None,
    client_assertion_source: TokenSource | None = None,
    authority: str = DEFAULT_ENTRA_AUTHORITY,
    token_exchange_audience: str = DEFAULT_TOKEN_EXCHANGE_AUDIENCE,
    expiry_buffer_seconds: int = DEFAULT_EXPIRY_BUFFER_SECONDS,
    cache_max_size: int = DEFAULT_CACHE_MAX_SIZE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> None
```

Initialize the source.

**Parameters:**

- **tenant_id** (<code>str</code>) – The Microsoft Entra tenant ID.
- **blueprint_client_id** (<code>str</code>) – The client ID of the agent identity blueprint.
- **agent_identity_client_id** (<code>str</code>) – The client ID of the agent identity that acts for the user.
- **scopes** (<code>list\[str\]</code>) – The scopes of the downstream resource to request, for example
  `["https://graph.microsoft.com/.default"]`. All scopes must belong to the same resource.
- **client_secret** (<code>Secret | None</code>) – The blueprint's client secret.
- **certificate** (<code>Secret | None</code>) – The blueprint's PEM-encoded X.509 certificate. Must be set together with `private_key`.
- **private_key** (<code>Secret | None</code>) – The PEM-encoded, unencrypted private key of `certificate`.
- **client_assertion_source** (<code>TokenSource | None</code>) – A config-only token source that yields a client assertion for the blueprint,
  for credentials not covered by the parameters above.
- **authority** (<code>str</code>) – The Microsoft Entra authority host. Change it for sovereign clouds.
- **token_exchange_audience** (<code>str</code>) – The token-exchange audience of the cloud. The US Government and China clouds
  use `api://AzureADTokenExchangeUSGov` and `api://AzureADTokenExchangeChina`.
- **expiry_buffer_seconds** (<code>int</code>) – Request a new token this many seconds before a cached one expires.
- **cache_max_size** (<code>int</code>) – The maximum number of per-user access tokens to keep in the in-memory cache. The
  least-recently-used entry is evicted when the cache is full.
- **timeout** (<code>float</code>) – The timeout, in seconds, for each request to the token endpoint.

**Raises:**

- <code>ImportError</code> – If a certificate is configured and the `jwt` extra is not installed.
- <code>OAuthConfigError</code> – If a required value is empty, or if the blueprint credential is missing or
  ambiguous.

#### resolve

```python
resolve(subject_token: str) -> str
```

Exchange the user's token for an access token to the configured resource (cached per user token).

**Parameters:**

- **subject_token** (<code>str</code>) – The signed-in user's access token, issued for the blueprint.

**Returns:**

- <code>str</code> – A valid bearer access token for the resource, acting on behalf of the user.

**Raises:**

- <code>OAuthConfigError</code> – If `subject_token` is empty.
- <code>TokenRefreshError</code> – If the blueprint credential cannot be used or Microsoft Entra rejects one of the
  two requests. The message carries Entra's error response, including its `AADSTS` code.

#### resolve_async

```python
resolve_async(subject_token: str) -> str
```

Asynchronous counterpart of `resolve`.

**Parameters:**

- **subject_token** (<code>str</code>) – The signed-in user's access token, issued for the blueprint.

**Returns:**

- <code>str</code> – A valid bearer access token for the resource, acting on behalf of the user.

**Raises:**

- <code>OAuthConfigError</code> – If `subject_token` is empty.
- <code>TokenRefreshError</code> – If the blueprint credential cannot be used or Microsoft Entra rejects one of the
  two requests. The message carries Entra's error response, including its `AADSTS` code.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the source to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> EntraAgentOBOTokenSource
```

Deserialize the source from a dictionary.

## haystack_integrations.utils.oauth.errors

### OAuthError

Bases: <code>Exception</code>

Base class for errors raised by the OAuth integration.

### OAuthConfigError

Bases: <code>OAuthError</code>

Raised when an OAuth component or token source is misconfigured.

### TokenRefreshError

Bases: <code>OAuthError</code>

Raised when a token cannot be resolved or refreshed at the identity provider.

## haystack_integrations.utils.oauth.protocols

### TokenSource

Bases: <code>Protocol</code>

A token source that resolves an access token with no per-request input (a config-only source).

Implemented by sources whose credential is fixed at construction time — e.g. `OAuthRefreshTokenSource`,
`OAuthClientCredentialsSource`, and `OAuthStaticTokenSource`. Such sources set the class attribute
`requires_subject_token = False`, and `OAuthTokenResolver` runs them as source nodes (no run input).

A config-only source can also supply the client assertion that authenticates another source: see the
`client_assertion_source` parameter of `OAuthClientCredentialsSource` and `OAuthTokenExchangeSource`, and
`OAuthPrivateKeyJWTSource`, which is built for that purpose.

#### resolve

```python
resolve() -> str
```

Return a valid access token.

#### resolve_async

```python
resolve_async() -> str
```

Asynchronous counterpart of `resolve`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the source to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> TokenSource
```

Deserialize the source from a dictionary.

### SubjectTokenSource

Bases: <code>Protocol</code>

A token source that resolves an access token by exchanging a per-request subject token.

The `subject_token` is a controller-injected per-request credential (for example an incoming user assertion),
not chosen by an end user. Implemented by `OAuthTokenExchangeSource` and `EntraAgentOBOTokenSource`. Such sources
set the class attribute `requires_subject_token = True`, which makes `OAuthTokenResolver` declare a mandatory
`subject_token` run input.

#### resolve

```python
resolve(subject_token: str) -> str
```

Return a valid access token for the per-request `subject_token`.

#### resolve_async

```python
resolve_async(subject_token: str) -> str
```

Asynchronous counterpart of `resolve`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the source to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SubjectTokenSource
```

Deserialize the source from a dictionary.

## haystack_integrations.utils.oauth.sources

### OAuthRefreshTokenSource

Resolves access tokens by running the RFC 6749 refresh-token grant against an OAuth token endpoint.

Given a stored refresh token plus client credentials, it exchanges them for an access token and caches it in
process until shortly before expiry. If the identity provider rotates the refresh token on exchange, the new value
is kept for the lifetime of the process and surfaced through the optional `on_rotate` callback so it can be
persisted.

This source is **single-identity**: one refresh token per instance, and its in-process cache is not shared across
processes. In a multi-replica deployment each replica keeps its own cache, so for providers that rotate (issue
single-use) refresh tokens the replicas can invalidate one another's token unless rotations are persisted to a
shared store via `on_rotate` and a single owner drives the refresh.

Choose this source for a single fixed identity backed by a refresh grant. For a long-lived, non-expiring token
use `OAuthStaticTokenSource`; for multi-replica or multi-user backends use `OAuthTokenExchangeSource`.

#### __init__

```python
__init__(
    token_url: str,
    client_id: str,
    *,
    refresh_token: Secret = Secret.from_env_var("OAUTH_REFRESH_TOKEN"),
    client_secret: Secret | None = None,
    scopes: list[str] | None = None,
    scope_delimiter: str = " ",
    expiry_buffer_seconds: int = DEFAULT_EXPIRY_BUFFER_SECONDS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    on_rotate: Callable[[str], None] | None = None
) -> None
```

Initialize the source.

**Parameters:**

- **token_url** (<code>str</code>) – The OAuth 2.0 token endpoint.
- **client_id** (<code>str</code>) – The OAuth client identifier.
- **refresh_token** (<code>Secret</code>) – The refresh token to exchange. Defaults to the value of the `OAUTH_REFRESH_TOKEN`
  environment variable.
- **client_secret** (<code>Secret | None</code>) – The client secret for confidential clients. Omit it for public clients.
- **scopes** (<code>list\[str\] | None</code>) – The OAuth scopes to request, joined with `scope_delimiter`. Scope *values* are
  provider-specific (consult your identity provider's documentation).
- **scope_delimiter** (<code>str</code>) – The delimiter used to join scopes. Defaults to a space (some providers use a comma).
- **expiry_buffer_seconds** (<code>int</code>) – Refresh the cached access token this many seconds before its declared expiry.
- **timeout** (<code>float</code>) – The timeout, in seconds, for the request to the token endpoint.
- **on_rotate** (<code>Callable\\[[str\], None\] | None</code>) – An optional callback invoked with the new refresh token whenever the provider rotates it.
  Use it to persist the rotated token durably (the source itself only keeps it in process).

**Raises:**

- <code>OAuthConfigError</code> – If the configuration is invalid.

#### resolve

```python
resolve() -> str
```

Return a cached access token, or run the refresh-token grant to obtain a fresh one.

**Returns:**

- <code>str</code> – A valid bearer access token.

**Raises:**

- <code>TokenRefreshError</code> – If the `refresh_token` or `client_secret` cannot be resolved, or if the token
  endpoint rejects the request or returns no access token.

#### resolve_async

```python
resolve_async() -> str
```

Asynchronous counterpart of `resolve`. Use a single instance in either sync or async mode, not both.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the source to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OAuthRefreshTokenSource
```

Deserialize the source from a dictionary.

### OAuthClientCredentialsSource

Resolves access tokens by running the RFC 6749 client-credentials grant against an OAuth token endpoint.

The token belongs to the client application itself, not to a user, so this source takes no per-request input.
Like `OAuthRefreshTokenSource` it is **single-identity**: one token per instance, cached in process until shortly
before expiry.

The client authenticates with a `client_secret` or with a client assertion (RFC 7523). The assertion is resolved
from another token source whenever a new token is requested: a JWT signed by `OAuthPrivateKeyJWTSource`, a fixed
value from `OAuthStaticTokenSource`, or a custom source.

Besides feeding an `OAuthTokenResolver`, an instance can be the `client_assertion_source` of
`OAuthTokenExchangeSource`, for providers where one client's token authenticates another client's exchange.

### Usage example

```python
from haystack.utils import Secret
from haystack_integrations.components.connectors.oauth import OAuthTokenResolver
from haystack_integrations.utils.oauth import OAuthClientCredentialsSource

resolver = OAuthTokenResolver(
    token_source=OAuthClientCredentialsSource(
        token_url="https://idp.example.com/oauth2/token",
        client_id="my-service",
        client_secret=Secret.from_env_var("OAUTH_CLIENT_SECRET"),
        scopes=["api.read"],
    ),
)
access_token = resolver.run()["access_token"]
```

#### __init__

```python
__init__(
    token_url: str,
    client_id: str,
    *,
    client_secret: Secret | None = None,
    client_assertion_source: TokenSource | None = None,
    client_assertion_type: str = DEFAULT_JWT_BEARER_CLIENT_ASSERTION_TYPE,
    scopes: list[str] | None = None,
    scope_delimiter: str = " ",
    extra_token_params: dict[str, str] | None = None,
    expiry_buffer_seconds: int = DEFAULT_EXPIRY_BUFFER_SECONDS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> None
```

Initialize the source.

**Parameters:**

- **token_url** (<code>str</code>) – The OAuth 2.0 token endpoint.
- **client_id** (<code>str</code>) – The OAuth client identifier.
- **client_secret** (<code>Secret | None</code>) – The client secret. Mutually exclusive with `client_assertion_source`.
- **client_assertion_source** (<code>TokenSource | None</code>) – A config-only token source whose token is sent as the `client_assertion`
  form parameter to authenticate the client, instead of a client secret. It is resolved each time a new
  access token is requested, so an expiring assertion stays fresh. Mutually exclusive with `client_secret`.
- **client_assertion_type** (<code>str</code>) – The value of the `client_assertion_type` form parameter sent along with the
  client assertion. Defaults to the RFC 7523 JWT bearer type.
- **scopes** (<code>list\[str\] | None</code>) – The OAuth scopes to request, joined with `scope_delimiter`. Scope *values* are
  provider-specific (consult your identity provider's documentation).
- **scope_delimiter** (<code>str</code>) – The delimiter used to join scopes. Defaults to a space.
- **extra_token_params** (<code>dict\[str, str\] | None</code>) – Additional form parameters included verbatim in every request. Applied last, so
  any key here overrides the corresponding form parameter derived from the other arguments (for example
  `scope`, `client_secret`, or `client_assertion`).
- **expiry_buffer_seconds** (<code>int</code>) – Request a new access token this many seconds before the cached one expires.
- **timeout** (<code>float</code>) – The timeout, in seconds, for the request to the token endpoint.

**Raises:**

- <code>OAuthConfigError</code> – If the configuration is invalid.

#### resolve

```python
resolve() -> str
```

Return a cached access token, or run the client-credentials grant to obtain a fresh one.

**Returns:**

- <code>str</code> – A valid bearer access token.

**Raises:**

- <code>TokenRefreshError</code> – If the `client_secret` cannot be resolved, if the `client_assertion_source` fails
  or returns an empty assertion, or if the token endpoint rejects the request or returns no access token.

#### resolve_async

```python
resolve_async() -> str
```

Asynchronous counterpart of `resolve`. Use a single instance in either sync or async mode, not both.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the source to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OAuthClientCredentialsSource
```

Deserialize the source from a dictionary.

### OAuthTokenExchangeSource

Resolves access tokens by exchanging a per-request subject token at an OAuth token endpoint.

This implements RFC 8693 token exchange (and, via configuration, Microsoft's on-behalf-of flow). Unlike
`OAuthRefreshTokenSource`, it is **multi-user without any persistent storage**: the per-request `subject_token` (the
incoming user assertion) *is* the user identity and is exchanged fresh for a downstream token. Resolved tokens
are cached in memory per subject token (bounded, LRU) until shortly before expiry. Because no per-instance state
is persisted, it is also the right choice for multi-replica deployments.

Provider differences are expressed as configuration: `grant_type`, `subject_token_param` (for example
`assertion` for Microsoft), `scopes`, and `extra_token_params` (for example
`{"requested_token_use": "on_behalf_of"}`).

The client authenticates with a `client_secret` or with a client assertion (RFC 7523) taken from
`client_assertion_source`: a JWT signed by `OAuthPrivateKeyJWTSource` for certificate authentication, or another
client's token from `OAuthClientCredentialsSource`. For the Microsoft Entra Agent ID on-behalf-of flow, which
needs the latter, use the preconfigured `EntraAgentOBOTokenSource`.

#### __init__

```python
__init__(
    token_url: str,
    client_id: str,
    *,
    client_secret: Secret | None = None,
    client_assertion_source: TokenSource | None = None,
    client_assertion_type: str = DEFAULT_JWT_BEARER_CLIENT_ASSERTION_TYPE,
    grant_type: str = DEFAULT_TOKEN_EXCHANGE_GRANT,
    subject_token_param: str = "subject_token",
    subject_token_type: str | None = None,
    requested_token_type: str | None = None,
    scopes: list[str] | None = None,
    scope_delimiter: str = " ",
    extra_token_params: dict[str, str] | None = None,
    expiry_buffer_seconds: int = DEFAULT_EXPIRY_BUFFER_SECONDS,
    cache_max_size: int = DEFAULT_CACHE_MAX_SIZE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> None
```

Initialize the source.

**Parameters:**

- **token_url** (<code>str</code>) – The OAuth 2.0 token endpoint.
- **client_id** (<code>str</code>) – The OAuth client identifier.
- **client_secret** (<code>Secret | None</code>) – The client secret for confidential clients. Omit it for public clients. Mutually
  exclusive with `client_assertion_source`.
- **client_assertion_source** (<code>TokenSource | None</code>) – A config-only token source whose token is sent as the `client_assertion`
  form parameter to authenticate the client, instead of a client secret. It is resolved for every exchange
  that is not served from the cache, so an expiring assertion stays fresh. Mutually exclusive with
  `client_secret`. A custom source must live in a module that Haystack is allowed to deserialize.
- **client_assertion_type** (<code>str</code>) – The value of the `client_assertion_type` form parameter sent along with the
  client assertion. Defaults to the RFC 7523 JWT bearer type.
- **grant_type** (<code>str</code>) – The grant type sent as the `grant_type` form parameter. Defaults to the RFC 8693
  token-exchange grant. Set it to the value your provider expects (for example the
  `urn:ietf:params:oauth:grant-type:jwt-bearer` grant for Microsoft on-behalf-of).
- **subject_token_param** (<code>str</code>) – The name of the form parameter carrying the per-request subject token. Defaults
  to `subject_token` (RFC 8693). Some providers expect a different name, such as `assertion`.
- **subject_token_type** (<code>str | None</code>) – The RFC 8693 identifier for the type of the supplied subject token, sent as the
  `subject_token_type` form parameter (omitted when not set). Required by RFC 8693 token exchange
  (e.g. `urn:ietf:params:oauth:token-type:access_token`); not used by Microsoft's on-behalf-of flow.
- **requested_token_type** (<code>str | None</code>) – The RFC 8693 identifier for the token to return, sent as the
  `requested_token_type` form parameter (omitted when not set). Optional.
- **scopes** (<code>list\[str\] | None</code>) – The OAuth scopes to request, joined with `scope_delimiter`. Scope *values* are
  provider-specific (consult your identity provider's documentation); only the wire format is standardized
  (RFC 6749 §3.3).
- **scope_delimiter** (<code>str</code>) – The delimiter used to join scopes. Defaults to a space.
- **extra_token_params** (<code>dict\[str, str\] | None</code>) – Additional form parameters included verbatim in every request (for example
  `{"requested_token_use": "on_behalf_of"}`). Applied last, so any key here overrides the corresponding
  form parameter derived from the other arguments (for example `grant_type`, `subject_token_type`,
  `requested_token_type`, `scope`, `client_secret`, or `client_assertion`).
- **expiry_buffer_seconds** (<code>int</code>) – Refresh a cached access token this many seconds before its declared expiry.
- **cache_max_size** (<code>int</code>) – The maximum number of per-user tokens to keep in the in-memory cache. The
  least-recently-used entry is evicted when the cache is full.
- **timeout** (<code>float</code>) – The timeout, in seconds, for the request to the token endpoint.

**Raises:**

- <code>OAuthConfigError</code> – If the configuration is invalid.

#### resolve

```python
resolve(subject_token: str) -> str
```

Exchange the per-request `subject_token` for an access token (cached per subject token).

**Parameters:**

- **subject_token** (<code>str</code>) – The controller-injected per-request subject token (for example an incoming user
  assertion) to exchange for a downstream access token.

**Returns:**

- <code>str</code> – A valid bearer access token for the given `subject_token`.

**Raises:**

- <code>OAuthConfigError</code> – If `subject_token` is empty.
- <code>TokenRefreshError</code> – If the `client_secret` cannot be resolved, if the `client_assertion_source` fails
  or returns an empty assertion, or if the token endpoint rejects the exchange or returns no access token.

#### resolve_async

```python
resolve_async(subject_token: str) -> str
```

Asynchronous counterpart of `resolve`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the source to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OAuthTokenExchangeSource
```

Deserialize the source from a dictionary.

### OAuthStaticTokenSource

Returns a configured long-lived access token as-is.

Suitable for providers that issue non-expiring tokens (for example Slack or Notion), where no refresh flow is
needed and the token is managed out of band. If the provider issues short-lived tokens that must be refreshed,
use `OAuthRefreshTokenSource` instead. It takes no per-request input.

#### __init__

```python
__init__(token: Secret) -> None
```

Initialize the source.

**Parameters:**

- **token** (<code>Secret</code>) – The long-lived access token to return.

#### resolve

```python
resolve() -> str
```

Return the configured token.

**Returns:**

- <code>str</code> – The configured long-lived access token.

**Raises:**

- <code>TokenRefreshError</code> – If the token cannot be resolved or is empty.

#### resolve_async

```python
resolve_async() -> str
```

Asynchronous counterpart of `resolve`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the source to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OAuthStaticTokenSource
```

Deserialize the source from a dictionary.

### OAuthPrivateKeyJWTSource

Signs a short-lived JWT client assertion (RFC 7523, also known as `private_key_jwt`) with a private key.

The result is not an access token: it proves the client's identity to a token endpoint. Pass an instance as the
`client_assertion_source` of `OAuthClientCredentialsSource` or `OAuthTokenExchangeSource` to authenticate with a
key pair or a certificate instead of a client secret.

A new assertion is signed on every call, so it is never stale. The private key must be an unencrypted PEM key;
passphrase-protected keys and PFX files are not supported.

Requires the `jwt` extra: `pip install "oauth-haystack[jwt]"`.

### Usage example

```python
from haystack.utils import Secret
from haystack_integrations.utils.oauth import OAuthClientCredentialsSource, OAuthPrivateKeyJWTSource

token_url = "https://idp.example.com/oauth2/token"
source = OAuthClientCredentialsSource(
    token_url=token_url,
    client_id="my-service",
    client_assertion_source=OAuthPrivateKeyJWTSource(
        client_id="my-service",
        audience=token_url,
        private_key=Secret.from_env_var("OAUTH_PRIVATE_KEY"),
    ),
    scopes=["api.read"],
)
access_token = source.resolve()
```

#### __init__

```python
__init__(
    client_id: str,
    audience: str,
    *,
    private_key: Secret = Secret.from_env_var("OAUTH_PRIVATE_KEY"),
    certificate: Secret | None = None,
    algorithm: str = "RS256",
    key_id: str | None = None,
    lifetime_seconds: int = DEFAULT_CLIENT_ASSERTION_LIFETIME_SECONDS
) -> None
```

Initialize the source.

**Parameters:**

- **client_id** (<code>str</code>) – The OAuth client identifier, used as the `iss` and `sub` claims of the assertion.
- **audience** (<code>str</code>) – The `aud` claim: the identifier the identity provider expects for itself, usually its token
  endpoint URL.
- **private_key** (<code>Secret</code>) – The PEM-encoded, unencrypted private key that signs the assertion. Defaults to the value
  of the `OAUTH_PRIVATE_KEY` environment variable.
- **certificate** (<code>Secret | None</code>) – The PEM-encoded X.509 certificate that belongs to the private key. When set, its SHA-256
  thumbprint is sent in the `x5t#S256` header (RFC 7515), which providers that register certificates
  rather than bare public keys use to find the key. If the PEM holds a chain, the first certificate is
  used.
- **algorithm** (<code>str</code>) – The JWT signing algorithm, for example `RS256`, `PS256`, or `ES256`. It must match the key
  type and be accepted by the identity provider.
- **key_id** (<code>str | None</code>) – The `kid` header value, for providers that select the verification key by key ID.
- **lifetime_seconds** (<code>int</code>) – How long each assertion is valid. Keep it short; identity providers commonly reject
  assertions that are valid for more than a few minutes.

**Raises:**

- <code>ImportError</code> – If the `jwt` extra is not installed.
- <code>OAuthConfigError</code> – If the configuration is invalid.

#### resolve

```python
resolve() -> str
```

Sign and return a new client assertion.

**Returns:**

- <code>str</code> – A signed JWT client assertion.

**Raises:**

- <code>TokenRefreshError</code> – If the private key or certificate cannot be resolved or parsed, or if the key
  cannot sign with the configured algorithm.

#### resolve_async

```python
resolve_async() -> str
```

Asynchronous counterpart of `resolve`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the source to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OAuthPrivateKeyJWTSource
```

Deserialize the source from a dictionary.
