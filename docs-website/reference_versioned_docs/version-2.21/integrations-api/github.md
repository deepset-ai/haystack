---
title: "GitHub"
id: integrations-github
description: "GitHub integration for Haystack"
slug: "/integrations-github"
---

<a id="haystack_integrations.components.connectors.github.file_editor"></a>

## Module haystack\_integrations.components.connectors.github.file\_editor

<a id="haystack_integrations.components.connectors.github.file_editor.Command"></a>

### Command

Available commands for file operations in GitHub.

**Attributes**:

- `EDIT` - Edit an existing file by replacing content
- `UNDO` - Revert the last commit if made by the same user
- `CREATE` - Create a new file
- `DELETE` - Delete an existing file

<a id="haystack_integrations.components.connectors.github.file_editor.GitHubFileEditor"></a>

### GitHubFileEditor

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

<a id="haystack_integrations.components.connectors.github.file_editor.GitHubFileEditor.__init__"></a>

#### GitHubFileEditor.\_\_init\_\_

```python
def __init__(*,
             github_token: Secret = Secret.from_env_var("GITHUB_TOKEN"),
             repo: Optional[str] = None,
             branch: str = "main",
             raise_on_failure: bool = True)
```

Initialize the component.

**Arguments**:

- `github_token`: GitHub personal access token for API authentication
- `repo`: Default repository in owner/repo format
- `branch`: Default branch to work with
- `raise_on_failure`: If True, raises exceptions on API errors

**Raises**:

- `TypeError`: If github_token is not a Secret

<a id="haystack_integrations.components.connectors.github.file_editor.GitHubFileEditor.run"></a>

#### GitHubFileEditor.run

```python
@component.output_types(result=str)
def run(command: Union[Command, str],
        payload: Dict[str, Any],
        repo: Optional[str] = None,
        branch: Optional[str] = None) -> Dict[str, str]
```

Process GitHub file operations.

**Arguments**:

- `command`: Operation to perform ("edit", "undo", "create", "delete")
- `payload`: Dictionary containing command-specific parameters
- `repo`: Repository in owner/repo format (overrides default if provided)
- `branch`: Branch to perform operations on (overrides default if provided)

**Raises**:

- `ValueError`: If command is not a valid Command enum value

**Returns**:

Dictionary containing operation result

<a id="haystack_integrations.components.connectors.github.file_editor.GitHubFileEditor.to_dict"></a>

#### GitHubFileEditor.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize the component to a dictionary.

<a id="haystack_integrations.components.connectors.github.file_editor.GitHubFileEditor.from_dict"></a>

#### GitHubFileEditor.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "GitHubFileEditor"
```

Deserialize the component from a dictionary.

<a id="haystack_integrations.components.connectors.github.issue_commenter"></a>

## Module haystack\_integrations.components.connectors.github.issue\_commenter

<a id="haystack_integrations.components.connectors.github.issue_commenter.GitHubIssueCommenter"></a>

### GitHubIssueCommenter

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

<a id="haystack_integrations.components.connectors.github.issue_commenter.GitHubIssueCommenter.__init__"></a>

#### GitHubIssueCommenter.\_\_init\_\_

```python
def __init__(*,
             github_token: Secret = Secret.from_env_var("GITHUB_TOKEN"),
             raise_on_failure: bool = True,
             retry_attempts: int = 2)
```

Initialize the component.

**Arguments**:

- `github_token`: GitHub personal access token for API authentication as a Secret
- `raise_on_failure`: If True, raises exceptions on API errors
- `retry_attempts`: Number of retry attempts for failed requests

<a id="haystack_integrations.components.connectors.github.issue_commenter.GitHubIssueCommenter.to_dict"></a>

#### GitHubIssueCommenter.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.connectors.github.issue_commenter.GitHubIssueCommenter.from_dict"></a>

#### GitHubIssueCommenter.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "GitHubIssueCommenter"
```

Deserialize the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.connectors.github.issue_commenter.GitHubIssueCommenter.run"></a>

#### GitHubIssueCommenter.run

```python
@component.output_types(success=bool)
def run(url: str, comment: str) -> dict
```

Post a comment to a GitHub issue.

**Arguments**:

- `url`: GitHub issue URL
- `comment`: Comment text to post

**Returns**:

Dictionary containing success status

<a id="haystack_integrations.components.connectors.github.issue_viewer"></a>

## Module haystack\_integrations.components.connectors.github.issue\_viewer

<a id="haystack_integrations.components.connectors.github.issue_viewer.GitHubIssueViewer"></a>

### GitHubIssueViewer

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

<a id="haystack_integrations.components.connectors.github.issue_viewer.GitHubIssueViewer.__init__"></a>

#### GitHubIssueViewer.\_\_init\_\_

```python
def __init__(*,
             github_token: Optional[Secret] = None,
             raise_on_failure: bool = True,
             retry_attempts: int = 2)
```

Initialize the component.

**Arguments**:

- `github_token`: GitHub personal access token for API authentication as a Secret
- `raise_on_failure`: If True, raises exceptions on API errors
- `retry_attempts`: Number of retry attempts for failed requests

<a id="haystack_integrations.components.connectors.github.issue_viewer.GitHubIssueViewer.to_dict"></a>

#### GitHubIssueViewer.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.connectors.github.issue_viewer.GitHubIssueViewer.from_dict"></a>

#### GitHubIssueViewer.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "GitHubIssueViewer"
```

Deserialize the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.connectors.github.issue_viewer.GitHubIssueViewer.run"></a>

#### GitHubIssueViewer.run

```python
@component.output_types(documents=List[Document])
def run(url: str) -> dict
```

Process a GitHub issue URL and return documents.

**Arguments**:

- `url`: GitHub issue URL

**Returns**:

Dictionary containing list of documents

<a id="haystack_integrations.components.connectors.github.pr_creator"></a>

## Module haystack\_integrations.components.connectors.github.pr\_creator

<a id="haystack_integrations.components.connectors.github.pr_creator.GitHubPRCreator"></a>

### GitHubPRCreator

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
    title="Fix issue `123`",
    body="This PR addresses issue `123`",
    branch="feature-branch",     # The branch in your fork with the changes
    base="main"                  # The branch in the original repo to merge into
)
```

<a id="haystack_integrations.components.connectors.github.pr_creator.GitHubPRCreator.__init__"></a>

#### GitHubPRCreator.\_\_init\_\_

```python
def __init__(*,
             github_token: Secret = Secret.from_env_var("GITHUB_TOKEN"),
             raise_on_failure: bool = True)
```

Initialize the component.

**Arguments**:

- `github_token`: GitHub personal access token for authentication (from the fork owner)
- `raise_on_failure`: If True, raises exceptions on API errors

<a id="haystack_integrations.components.connectors.github.pr_creator.GitHubPRCreator.run"></a>

#### GitHubPRCreator.run

```python
@component.output_types(result=str)
def run(issue_url: str,
        title: str,
        branch: str,
        base: str,
        body: str = "",
        draft: bool = False) -> Dict[str, str]
```

Create a new pull request from your fork to the original repository, linked to the specified issue.

**Arguments**:

- `issue_url`: URL of the GitHub issue to link the PR to
- `title`: Title of the pull request
- `branch`: Name of the branch in your fork where changes are implemented
- `base`: Name of the branch in the original repo you want to merge into
- `body`: Additional content for the pull request description
- `draft`: Whether to create a draft pull request

**Returns**:

Dictionary containing operation result

<a id="haystack_integrations.components.connectors.github.pr_creator.GitHubPRCreator.to_dict"></a>

#### GitHubPRCreator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize the component to a dictionary.

<a id="haystack_integrations.components.connectors.github.pr_creator.GitHubPRCreator.from_dict"></a>

#### GitHubPRCreator.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "GitHubPRCreator"
```

Deserialize the component from a dictionary.

<a id="haystack_integrations.components.connectors.github.repo_viewer"></a>

## Module haystack\_integrations.components.connectors.github.repo\_viewer

<a id="haystack_integrations.components.connectors.github.repo_viewer.GitHubItem"></a>

### GitHubItem

Represents an item (file or directory) in a GitHub repository

<a id="haystack_integrations.components.connectors.github.repo_viewer.GitHubItem.type"></a>

#### type

"file" or "dir"

<a id="haystack_integrations.components.connectors.github.repo_viewer.GitHubRepoViewer"></a>

### GitHubRepoViewer

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

<a id="haystack_integrations.components.connectors.github.repo_viewer.GitHubRepoViewer.__init__"></a>

#### GitHubRepoViewer.\_\_init\_\_

```python
def __init__(*,
             github_token: Optional[Secret] = None,
             raise_on_failure: bool = True,
             max_file_size: int = 1_000_000,
             repo: Optional[str] = None,
             branch: str = "main")
```

Initialize the component.

**Arguments**:

- `github_token`: GitHub personal access token for API authentication
- `raise_on_failure`: If True, raises exceptions on API errors
- `max_file_size`: Maximum file size in bytes to fetch (default: 1MB)
- `repo`: Repository in format "owner/repo"
- `branch`: Git reference (branch, tag, commit) to use

<a id="haystack_integrations.components.connectors.github.repo_viewer.GitHubRepoViewer.to_dict"></a>

#### GitHubRepoViewer.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.connectors.github.repo_viewer.GitHubRepoViewer.from_dict"></a>

#### GitHubRepoViewer.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "GitHubRepoViewer"
```

Deserialize the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.connectors.github.repo_viewer.GitHubRepoViewer.run"></a>

#### GitHubRepoViewer.run

```python
@component.output_types(documents=List[Document])
def run(path: str,
        repo: Optional[str] = None,
        branch: Optional[str] = None) -> Dict[str, List[Document]]
```

Process a GitHub repository path and return documents.

**Arguments**:

- `repo`: Repository in format "owner/repo"
- `path`: Path within repository (default: root)
- `branch`: Git reference (branch, tag, commit) to use

**Returns**:

Dictionary containing list of documents

<a id="haystack_integrations.components.connectors.github.repo_forker"></a>

## Module haystack\_integrations.components.connectors.github.repo\_forker

<a id="haystack_integrations.components.connectors.github.repo_forker.GitHubRepoForker"></a>

### GitHubRepoForker

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

<a id="haystack_integrations.components.connectors.github.repo_forker.GitHubRepoForker.__init__"></a>

#### GitHubRepoForker.\_\_init\_\_

```python
def __init__(*,
             github_token: Secret = Secret.from_env_var("GITHUB_TOKEN"),
             raise_on_failure: bool = True,
             wait_for_completion: bool = False,
             max_wait_seconds: int = 300,
             poll_interval: int = 2,
             auto_sync: bool = True,
             create_branch: bool = True)
```

Initialize the component.

**Arguments**:

- `github_token`: GitHub personal access token for API authentication
- `raise_on_failure`: If True, raises exceptions on API errors
- `wait_for_completion`: If True, waits until fork is fully created
- `max_wait_seconds`: Maximum time to wait for fork completion in seconds
- `poll_interval`: Time between status checks in seconds
- `auto_sync`: If True, syncs fork with original repository if it already exists
- `create_branch`: If True, creates a fix branch based on the issue number

<a id="haystack_integrations.components.connectors.github.repo_forker.GitHubRepoForker.to_dict"></a>

#### GitHubRepoForker.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.connectors.github.repo_forker.GitHubRepoForker.from_dict"></a>

#### GitHubRepoForker.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "GitHubRepoForker"
```

Deserialize the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.connectors.github.repo_forker.GitHubRepoForker.run"></a>

#### GitHubRepoForker.run

```python
@component.output_types(repo=str, issue_branch=str)
def run(url: str) -> dict
```

Process a GitHub issue URL and create or sync a fork of the repository.

**Arguments**:

- `url`: GitHub issue URL

**Returns**:

Dictionary containing repository path in owner/repo format

