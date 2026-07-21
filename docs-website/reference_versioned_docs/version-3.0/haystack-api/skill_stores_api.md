---
title: "Skill Stores"
id: skill-stores-api
description: "Storage layers that discover skills and serve their content on demand."
slug: "/skill-stores-api"
---


## file_system/skill_store

### FileSystemSkillStore

SkillStore backed by a directory of skill sub-directories on the local filesystem.

Expected layout:

```
skills/
  pdf-forms/
    SKILL.md            # frontmatter (name, description) + markdown instructions
    reference/forms.md  # optional bundled file
```

The skill catalog is built by reading the frontmatter of each `SKILL.md` on `warm_up`; bodies and bundled files
are read lazily when the agent calls the corresponding tool.

#### __init__

```python
__init__(skills_dir: str | Path) -> None
```

Initialize the store with the root directory to scan.

No filesystem access happens here; the directory is scanned lazily on first use (see `warm_up`), so the store
can be constructed cheaply.

**Parameters:**

- **skills_dir** (<code>str | Path</code>) – Root directory that contains one sub-directory per skill.

#### warm_up

```python
warm_up() -> None
```

Scan `skills_dir` and build the skill catalog by reading each skill's `SKILL.md` frontmatter.

Only the frontmatter is read here; bodies and bundled files are read lazily when the corresponding method is
called. Idempotent: repeated calls after the first are no-ops.

**Raises:**

- <code>ValueError</code> – If `skills_dir` does not exist, is not a directory, a skill's frontmatter is missing,
  malformed, or missing a required field, or two skills share the same name.

#### list_skills

```python
list_skills() -> dict[str, SkillInfo]
```

Return all skills discovered on disk, warming up the store first if needed.

**Returns:**

- <code>dict\[str, SkillInfo\]</code> – Mapping of skill name to its metadata.

**Raises:**

- <code>ValueError</code> – If the skills directory is invalid or a skill's frontmatter is malformed.

#### load_skill

```python
load_skill(name: str) -> tuple[str, list[str]]
```

Read the named skill's instruction body and the manifest of its bundled files.

**Parameters:**

- **name** (<code>str</code>) – Skill name as returned by `list_skills`.

**Returns:**

- <code>tuple\[str, list\[str\]\]</code> – A tuple of (markdown body of the skill's `SKILL.md` with frontmatter stripped, sorted list of
  POSIX-style paths relative to the skill directory for any bundled files). The file list is empty when
  the skill bundles no extras.

**Raises:**

- <code>KeyError</code> – If no skill with `name` exists.

#### read_skill_file

```python
read_skill_file(name: str, path: str) -> str | ImageContent | FileContent
```

Read a file bundled with the named skill, preventing path traversal outside the skill directory.

The return type depends on the file: text files are returned as a `str`, image files (PNG, JPEG, ...) as an
`ImageContent`, and PDFs as a `FileContent`, so a multimodal agent can pass them straight to the model.

**Parameters:**

- **name** (<code>str</code>) – Skill name as returned by `list_skills`.
- **path** (<code>str</code>) – Path of the file relative to the skill directory (e.g. `"reference/forms.md"`).

**Returns:**

- <code>str | ImageContent | FileContent</code> – The file's text content (`str`), an `ImageContent` for images, or a `FileContent` for PDFs.

**Raises:**

- <code>KeyError</code> – If no skill with `name` exists.
- <code>PermissionError</code> – If `path` resolves outside the skill's directory (path-traversal attempt). The
  message lists the readable files so the caller can retry with a valid path.
- <code>FileNotFoundError</code> – If the file does not exist within the skill. The message lists the readable
  files so the caller can retry with a valid path.
- <code>ValueError</code> – If the file is binary but not a supported image or PDF (i.e. not UTF-8 text either).

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this store to a dictionary for use with `from_dict`.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary representation of the store.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> FileSystemSkillStore
```

Deserialize a `FileSystemSkillStore` from its dictionary representation.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary representation of the store, as produced by `to_dict`.

**Returns:**

- <code>FileSystemSkillStore</code> – A new `FileSystemSkillStore` instance.

## types/protocol

### SkillStore

Bases: <code>Protocol</code>

Protocol for a skill storage layer.

A `SkillStore` is responsible for discovering available skills and providing their content on demand. Implement
this protocol to back a `haystack.tools.SkillToolset` with any storage system — a local directory, a database,
a remote API, or an in-memory fixture.

Skills are identified by their `name`, which must be unique within a store. The `name` is the lookup key for every
method below; implementations resolve it to their own internal locator (a directory, a row id, an object key, ...).

Implementations may defer all I/O (filesystem reads, database connections, ...) until a method is actually called,
so a store can be constructed cheaply and only touch its backend on first use.

Skill content is text: instruction bodies and bundled files are returned as strings. Binary assets (images,
fonts, ...) are not supported.

#### list_skills

```python
list_skills() -> dict[str, SkillInfo]
```

Discover and return all available skills.

**Returns:**

- <code>dict\[str, SkillInfo\]</code> – Mapping of skill name to its metadata.

#### load_skill

```python
load_skill(name: str) -> tuple[str, list[str]]
```

Return the named skill's instruction body and the manifest of its bundled files.

**Parameters:**

- **name** (<code>str</code>) – Skill name as returned by `list_skills`.

**Returns:**

- <code>tuple\[str, list\[str\]\]</code> – A tuple of (markdown body with frontmatter stripped, sorted list of POSIX-style paths relative
  to the skill root for any bundled files). The file list is empty when the skill bundles no extras.

**Raises:**

- <code>KeyError</code> – If no skill with `name` exists.

#### read_skill_file

```python
read_skill_file(name: str, path: str) -> str | ImageContent | FileContent
```

Read a file bundled with the named skill.

Implementations should return text files as a `str`, image files as an `ImageContent`, and PDFs as a
`FileContent`, so a multimodal agent can pass binary assets straight to the model.

**Parameters:**

- **name** (<code>str</code>) – Skill name as returned by `list_skills`.
- **path** (<code>str</code>) – Path of the file relative to the skill root (e.g. `"reference/forms.md"`).

**Returns:**

- <code>str | ImageContent | FileContent</code> – The file's text content (`str`), an `ImageContent` for images, or a `FileContent` for PDFs.

**Raises:**

- <code>KeyError</code> – If no skill with `name` exists.
- <code>FileNotFoundError</code> – If the file does not exist within the skill.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this store to a dictionary for use with `from_dict`.

Implement both this method and `from_dict` to make your custom store serializable.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SkillStore
```

Deserialize a store from a dictionary produced by `to_dict`.

Implement both this method and `to_dict` to make your custom store serializable.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary as produced by `to_dict`.
