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
a standalone OAuth refresh grant (`OAuthRefreshTokenSource`), a per-request token exchange
(`OAuthTokenExchangeSource`), a static long-lived token (`OAuthStaticTokenSource`), or a custom source you
provide. A downstream component (for
example a SharePoint or Google Drive retriever) consumes the token via a normal connection and never knows how
it was resolved.

The run input depends on the token source. A source that needs a per-request credential (it sets
`requires_subject_token = True`, like `OAuthTokenExchangeSource`) makes the resolver declare a **mandatory**
`subject_token` input — a controller-injected per-request credential (for example an incoming user assertion),
not chosen by an end user. A config-only source declares no run input, so the resolver is a source node.

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
run(**kwargs: Any) -> dict[str, str]
```

Resolve an access token and emit it.

**Parameters:**

- **kwargs** (<code>Any</code>) – Carries `subject_token` when the configured source requires it (declared as a mandatory
  input in that case, injected by the application/controller per request). For config-only sources no
  input is declared and `kwargs` is empty.

**Returns:**

- <code>dict\[str, str\]</code> – A dictionary with a single `access_token` key containing a bearer token string.

**Raises:**

- <code>OAuthConfigError</code> – If the source requires a `subject_token` but it is missing or empty.

#### run_async

```python
run_async(**kwargs: Any) -> dict[str, str]
```

Asynchronously resolve an access token and emit it.

**Parameters:**

- **kwargs** (<code>Any</code>) – Carries `subject_token` when the configured source requires it.

**Returns:**

- <code>dict\[str, str\]</code> – A dictionary with a single `access_token` key containing a bearer token string.

**Raises:**

- <code>OAuthConfigError</code> – If the source requires a `subject_token` but it is missing or empty.

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

Implemented by sources whose credential is fixed at construction time — e.g. `OAuthRefreshTokenSource` and
`OAuthStaticTokenSource`. Such sources set the class attribute `requires_subject_token = False`, and
`OAuthTokenResolver` runs them as source nodes (no run input).

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
not chosen by an end user. Implemented by `OAuthTokenExchangeSource`. Such sources set the class attribute
`requires_subject_token = True`, which makes `OAuthTokenResolver` declare a mandatory `subject_token` run input.

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

#### __init__

```python
__init__(
    token_url: str,
    client_id: str,
    *,
    client_secret: Secret | None = None,
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
- **client_secret** (<code>Secret | None</code>) – The client secret for confidential clients. Omit it for public clients.
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
  `requested_token_type`, `scope`, or `client_secret`).
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
