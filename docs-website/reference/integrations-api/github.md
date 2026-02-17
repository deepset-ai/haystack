---
title: "GitHub"
id: integrations-github
description: "GitHub integration for Haystack"
slug: "/integrations-github"
---


## `haystack_integrations.components.connectors.github.file_editor`

### `Command`

Bases: <code>str</code>, <code>Enum</code>

Available commands for file operations in GitHub.

Attributes:
EDIT: Edit an existing file by replacing content
UNDO: Revert the last commit if made by the same user
CREATE: Create a new file
DELETE: Delete an existing file

### `GitHubFileEditor`

A Haystack component for editing files in GitHub repositories.

Supports editing, undoing changes, deleting files, and creating new files
through the GitHub API.

### Usage example

```python
from haystack_integrations.components.connectors.github import Command, GitHubFileEditor
from haystack.utils import Secret

# Initialize with default repo and branch
editor = GitHubFileEditor(
    github_token=Secret.from_env_var("GITHUB_TOKEN"),
    repo="owner/repo",
    branch="main"
)

# Edit a file using default repo and branch
result = editor.run(
    command=Command.EDIT,
    payload={
        "path": "path/to/file.py",
        "original": "def old_function():",
        "replacement": "def new_function():",
        "message": "Renamed function for clarity"
    }
)

# Edit a file in a different repo/branch
result = editor.run(
    command=Command.EDIT,
    repo="other-owner/other-repo",  # Override default repo
    branch="feature",  # Override default branch
    payload={
        "path": "path/to/file.py",
        "original": "def old_function():",
        "replacement": "def new_function():",
        "message": "Renamed function for clarity"
    }
)
```

#### `__init__`

```python
__init__(
    *,
    github_token: Secret = Secret.from_env_var("GITHUB_TOKEN"),
    repo: str | None = None,
    branch: str = "main",
    raise_on_failure: bool = True
)
```

Initialize the component.

**Parameters:**

- **github_token** (<code>Secret</code>) – GitHub personal access token for API authentication
- **repo** (<code>str | None</code>) – Default repository in owner/repo format
- **branch** (<code>str</code>) – Default branch to work with
- **raise_on_failure** (<code>bool</code>) – If True, raises exceptions on API errors

**Raises:**

- <code>TypeError</code> – If github_token is not a Secret

#### `run`

```python
run(
    command: Command | str,
    payload: dict[str, Any],
    repo: str | None = None,
    branch: str | None = None,
) -> dict[str, str]
```

Process GitHub file operations.

**Parameters:**

- **command** (<code>Command | str</code>) – Operation to perform ("edit", "undo", "create", "delete")
- **payload** (<code>dict\[str, Any\]</code>) – Dictionary containing command-specific parameters
- **repo** (<code>str | None</code>) – Repository in owner/repo format (overrides default if provided)
- **branch** (<code>str | None</code>) – Branch to perform operations on (overrides default if provided)

**Returns:**

- <code>dict\[str, str\]</code> – Dictionary containing operation result

**Raises:**

- <code>ValueError</code> – If command is not a valid Command enum value

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> GitHubFileEditor
```

Deserialize the component from a dictionary.

## `haystack_integrations.components.connectors.github.issue_commenter`

### `GitHubIssueCommenter`

Posts comments to GitHub issues.

The component takes a GitHub issue URL and comment text, then posts the comment
to the specified issue using the GitHub API.

### Usage example

```python
from haystack_integrations.components.connectors.github import GitHubIssueCommenter
from haystack.utils import Secret

commenter = GitHubIssueCommenter(github_token=Secret.from_env_var("GITHUB_TOKEN"))
result = commenter.run(
    url="https://github.com/owner/repo/issues/123",
    comment="Thanks for reporting this issue! We'll look into it."
)

print(result["success"])
```

#### `__init__`

```python
__init__(
    *,
    github_token: Secret = Secret.from_env_var("GITHUB_TOKEN"),
    raise_on_failure: bool = True,
    retry_attempts: int = 2
)
```

Initialize the component.

**Parameters:**

- **github_token** (<code>Secret</code>) – GitHub personal access token for API authentication as a Secret
- **raise_on_failure** (<code>bool</code>) – If True, raises exceptions on API errors
- **retry_attempts** (<code>int</code>) – Number of retry attempts for failed requests

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> GitHubIssueCommenter
```

Deserialize the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>GitHubIssueCommenter</code> – Deserialized component.

#### `run`

```python
run(url: str, comment: str) -> dict
```

Post a comment to a GitHub issue.

**Parameters:**

- **url** (<code>str</code>) – GitHub issue URL
- **comment** (<code>str</code>) – Comment text to post

**Returns:**

- <code>dict</code> – Dictionary containing success status

## `haystack_integrations.components.connectors.github.issue_viewer`

### `GitHubIssueViewer`

Fetches and parses GitHub issues into Haystack documents.

The component takes a GitHub issue URL and returns a list of documents where:

- First document contains the main issue content
- Subsequent documents contain the issue comments

### Usage example

```python
from haystack_integrations.components.connectors.github import GitHubIssueViewer

viewer = GitHubIssueViewer()
docs = viewer.run(
    url="https://github.com/owner/repo/issues/123"
)["documents"]

print(docs)
```

#### `__init__`

```python
__init__(
    *,
    github_token: Secret | None = None,
    raise_on_failure: bool = True,
    retry_attempts: int = 2
)
```

Initialize the component.

**Parameters:**

- **github_token** (<code>Secret | None</code>) – GitHub personal access token for API authentication as a Secret
- **raise_on_failure** (<code>bool</code>) – If True, raises exceptions on API errors
- **retry_attempts** (<code>int</code>) – Number of retry attempts for failed requests

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> GitHubIssueViewer
```

Deserialize the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>GitHubIssueViewer</code> – Deserialized component.

#### `run`

```python
run(url: str) -> dict
```

Process a GitHub issue URL and return documents.

**Parameters:**

- **url** (<code>str</code>) – GitHub issue URL

**Returns:**

- <code>dict</code> – Dictionary containing list of documents

## `haystack_integrations.components.connectors.github.pr_creator`

### `GitHubPRCreator`

A Haystack component for creating pull requests from a fork back to the original repository.

Uses the authenticated user's fork to create the PR and links it to an existing issue.

### Usage example

```python
from haystack_integrations.components.connectors.github import GitHubPRCreator
from haystack.utils import Secret

pr_creator = GitHubPRCreator(
    github_token=Secret.from_env_var("GITHUB_TOKEN")  # Token from the fork owner
)

# Create a PR from your fork
result = pr_creator.run(
    issue_url="https://github.com/owner/repo/issues/123",
    title="Fix issue #123",
    body="This PR addresses issue #123",
    branch="feature-branch",     # The branch in your fork with the changes
    base="main"                  # The branch in the original repo to merge into
)
```

#### `__init__`

```python
__init__(
    *,
    github_token: Secret = Secret.from_env_var("GITHUB_TOKEN"),
    raise_on_failure: bool = True
)
```

Initialize the component.

**Parameters:**

- **github_token** (<code>Secret</code>) – GitHub personal access token for authentication (from the fork owner)
- **raise_on_failure** (<code>bool</code>) – If True, raises exceptions on API errors

#### `run`

```python
run(
    issue_url: str,
    title: str,
    branch: str,
    base: str,
    body: str = "",
    draft: bool = False,
) -> dict[str, str]
```

Create a new pull request from your fork to the original repository, linked to the specified issue.

**Parameters:**

- **issue_url** (<code>str</code>) – URL of the GitHub issue to link the PR to
- **title** (<code>str</code>) – Title of the pull request
- **branch** (<code>str</code>) – Name of the branch in your fork where changes are implemented
- **base** (<code>str</code>) – Name of the branch in the original repo you want to merge into
- **body** (<code>str</code>) – Additional content for the pull request description
- **draft** (<code>bool</code>) – Whether to create a draft pull request

**Returns:**

- <code>dict\[str, str\]</code> – Dictionary containing operation result

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> GitHubPRCreator
```

Deserialize the component from a dictionary.

## `haystack_integrations.components.connectors.github.repo_forker`

### `GitHubRepoForker`

Forks a GitHub repository from an issue URL.

The component takes a GitHub issue URL, extracts the repository information,
creates or syncs a fork of that repository, and optionally creates an issue-specific branch.

### Usage example

```python
from haystack_integrations.components.connectors.github import GitHubRepoForker
from haystack.utils import Secret

# Using direct token with auto-sync and branch creation
forker = GitHubRepoForker(
    github_token=Secret.from_env_var("GITHUB_TOKEN"),
    auto_sync=True,
    create_branch=True
)

result = forker.run(url="https://github.com/owner/repo/issues/123")
print(result)
# Will create or sync fork and create branch "fix-123"
```

#### `__init__`

```python
__init__(
    *,
    github_token: Secret = Secret.from_env_var("GITHUB_TOKEN"),
    raise_on_failure: bool = True,
    wait_for_completion: bool = False,
    max_wait_seconds: int = 300,
    poll_interval: int = 2,
    auto_sync: bool = True,
    create_branch: bool = True
)
```

Initialize the component.

**Parameters:**

- **github_token** (<code>Secret</code>) – GitHub personal access token for API authentication
- **raise_on_failure** (<code>bool</code>) – If True, raises exceptions on API errors
- **wait_for_completion** (<code>bool</code>) – If True, waits until fork is fully created
- **max_wait_seconds** (<code>int</code>) – Maximum time to wait for fork completion in seconds
- **poll_interval** (<code>int</code>) – Time between status checks in seconds
- **auto_sync** (<code>bool</code>) – If True, syncs fork with original repository if it already exists
- **create_branch** (<code>bool</code>) – If True, creates a fix branch based on the issue number

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> GitHubRepoForker
```

Deserialize the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>GitHubRepoForker</code> – Deserialized component.

#### `run`

```python
run(url: str) -> dict
```

Process a GitHub issue URL and create or sync a fork of the repository.

**Parameters:**

- **url** (<code>str</code>) – GitHub issue URL

**Returns:**

- <code>dict</code> – Dictionary containing repository path in owner/repo format

## `haystack_integrations.components.connectors.github.repo_viewer`

### `GitHubItem`

Represents an item (file or directory) in a GitHub repository

### `GitHubRepoViewer`

Navigates and fetches content from GitHub repositories.

For directories:

- Returns a list of Documents, one for each item
- Each Document's content is the item name
- Full path and metadata in Document.meta

For files:

- Returns a single Document
- Document's content is the file content
- Full path and metadata in Document.meta

For errors:

- Returns a single Document
- Document's content is the error message
- Document's meta contains type="error"

### Usage example

```python
from haystack_integrations.components.connectors.github import GitHubRepoViewer

viewer = GitHubRepoViewer()

# List directory contents - returns multiple documents
result = viewer.run(
    repo="owner/repository",
    path="docs/",
    branch="main"
)
print(result)

# Get specific file - returns single document
result = viewer.run(
    repo="owner/repository",
    path="README.md",
    branch="main"
)
print(result)
```

#### `__init__`

```python
__init__(
    *,
    github_token: Secret | None = None,
    raise_on_failure: bool = True,
    max_file_size: int = 1000000,
    repo: str | None = None,
    branch: str = "main"
)
```

Initialize the component.

**Parameters:**

- **github_token** (<code>Secret | None</code>) – GitHub personal access token for API authentication
- **raise_on_failure** (<code>bool</code>) – If True, raises exceptions on API errors
- **max_file_size** (<code>int</code>) – Maximum file size in bytes to fetch (default: 1MB)
- **repo** (<code>str | None</code>) – Repository in format "owner/repo"
- **branch** (<code>str</code>) – Git reference (branch, tag, commit) to use

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> GitHubRepoViewer
```

Deserialize the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>GitHubRepoViewer</code> – Deserialized component.

#### `run`

```python
run(
    path: str, repo: str | None = None, branch: str | None = None
) -> dict[str, list[Document]]
```

Process a GitHub repository path and return documents.

**Parameters:**

- **repo** (<code>str | None</code>) – Repository in format "owner/repo"
- **path** (<code>str</code>) – Path within repository (default: root)
- **branch** (<code>str | None</code>) – Git reference (branch, tag, commit) to use

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – Dictionary containing list of documents
