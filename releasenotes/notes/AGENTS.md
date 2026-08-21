<!-- Mined from deepset PR reviews; see the repo-root AGENTS.md. -->

# releasenotes/notes/ Guidelines

## Documentation

- Add `upgrade` notes for breaking/user-visible changes in `releasenotes/notes/` — explain affected users, old/new behavior, and migration steps
- Add `releasenotes/notes/` entries only for in-scope user-facing PR changes — keeps release notes accurate and low-noise; leave unrelated note files untouched.
- Name affected APIs/configs in `releasenotes/notes/` — clarifies scope and impact for users
- Write bug/security notes around public impact, not private helpers — clarifies user risk
- Highlight APIs in `releasenotes/notes/` only with examples or clear use cases — shows practical value
- Keep `releasenotes/notes/` reno notes synced with shipped behavior — prevents misleading release docs
- Use one `releasenotes/notes/` file per PR — group related change notes together
- Proofread `releasenotes/notes/` entries — catches API typos and formatting issues
