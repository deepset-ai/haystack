# releasenotes/notes/ Guidelines

## Documentation

- Add `upgrade` notes for breaking/user-visible changes in `releasenotes/notes/` — explain affected users, old/new behavior, and migration steps
- Write one concise, user-facing note file per PR, only for in-scope changes and only under sections from `releasenotes/config.yaml`; name the affected APIs/configs and the old and new behavior, describe impact in user terms rather than private helpers, and leave unrelated note files untouched
- Highlight APIs in `releasenotes/notes/` only with examples or clear use cases — shows practical value
- Keep notes synced with shipped behavior and proofread them — API names, and reStructuredText formatting with double backticks for inline code
