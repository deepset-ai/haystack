# Plan: Skills support for the Haystack Agent

Goal: let the Haystack `Agent` discover and read filesystem **Skills** the way Claude
Code / Codex do — progressive disclosure of expert instructions — using a small set of
pre-built tools plus a generic mechanism for tools to contribute system-prompt text.

## Background — what we're replicating

Claude Code / Codex Skills use **progressive disclosure**:

| Level | What's loaded | When |
|---|---|---|
| 1 — Metadata | each skill's `name` + `description` | always, up front |
| 2 — Instructions | the full `SKILL.md` body | when a skill is triggered |
| 3 — Bundled files | `reference/*.md`, examples, etc. | on demand, as the body references them |

A skill is a directory:

```
skills/
  pdf-forms/
    SKILL.md            # YAML frontmatter (name, description) + markdown body
    reference/forms.md
```

Level 1 lives in the system context (not a tool). Levels 2/3 are pulled in on demand.

## Decisions (locked)

- **Two tools**, both pure/stateless: `load_skill`, `read_skill_file`.
- **No script execution** in v1 (no Level-3 execution tool).
- **Filesystem source only.**
- **System-prompt injection:** a generic contribution hook on `Tool`/`Toolset`,
  consumed automatically by the `Agent`.
  - An alternative to this approach is to add another tool that lists available skills and their descriptions and
    force it to be called using the soon to be added condition param.

## How MCP does the equivalent (reference)

MCP servers return an optional top-level `instructions` string in their `initialize` response.
The spec frames it as a hint clients MAY add to the system prompt to explain the server's tools as a whole.
Advantage of adding the **System-prompt injection** feature is that we could extend MCPToolset to inject these
top-level instructions.

## Part 1 — `system_prompt_contribution()` hook (generic, reusable)

Add an optional method to both base classes, default `None`:

TODO: Add support for ComponentTool + PipelineTool

```python
# haystack/tools/tool.py  (on Tool)
# haystack/tools/toolset.py  (on Toolset)
def system_prompt_contribution(self) -> str | None:
    """Text this tool/toolset wants appended to the Agent's system prompt. None by default."""
    return None
```

- `Tool` also gains an optional `system_prompt: str | None = None` dataclass field so a plain `Tool` can carry
  instructions without subclassing. `system_prompt_contribution()` returns it by default (subclasses may override for
  dynamism).
- `Toolset` subclasses (like `SkillToolset`) override the method.

### Agent consumption — `_initialize_fresh_execution`

Right after `selected_tools = self._select_tools(tools)`, collect contributions and merge them into the system message:

```python
selected_tools = self._select_tools(tools)
contributions = _collect_system_prompt_contributions(selected_tools)
if contributions:
    messages = _merge_system_prompt_contributions(messages, contributions)
```

Collection rules:

- Go through `list`, `_ToolsetWrapper` (descend into `.toolsets`), `Toolset`, `Tool`.
- **Top-level wins:** if a `Toolset` returns a contribution, use it and DO NOT descend into
  its member tools. Only if it returns `None` do we gather member tools' contributions.
- Bare `Tool` passed directly contributes its own.

Merge rules:

- Join contributions with `\n\n`.
- If `messages[0]` is a system message (the rendered `system_prompt`, or a user-supplied system message), append
  the contribution text to it.
- Otherwise prepend a new system message built from the contributions.
- Collect from `selected_tools` and inject **after** Jinja rendering — never into the template string — so skill text
  containing `{{`/`{%` can't break `ChatPromptBuilder`.

## Part 2 — `SkillToolset(Toolset)`

`haystack/tools/skills/skill_toolset.py`

```python
@dataclass
class SkillMeta:
    name: str           # from frontmatter; falls back to directory name
    description: str     # from frontmatter
    path: Path           # the skill directory

class SkillToolset(Toolset):
    def __init__(self, skills_dir: str | Path) -> None:
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, SkillMeta] = self._scan()   # frontmatter-only, cheap, in __init__
        super().__init__(tools=[self._load_skill_tool(), self._read_skill_file_tool()])
```

- `_scan()` walks `skills_dir/*/SKILL.md`, parses YAML frontmatter (pyyaml — already a
  dep), validates `name`/`description`, checks name uniqueness. Bodies are NOT read here.
- `system_prompt_contribution()` renders the Level-1 catalog + behavioral rules (below).
- `warm_up()` revalidates (idempotent).
- `to_dict()`/`from_dict()` serialize `skills_dir` only and rescan on load (mirrors
  `SearchableToolset`). `add`/`__add__` of new ad-hoc tools left as default Toolset behavior.

### Tool: `load_skill` (Level 2)

```python
def load_skill(name: Annotated[str, "Exact skill name from the Available Skills list."]) -> str:
    """Load a skill's full instructions. Call this before doing a task the skill covers."""
```

Returns the `SKILL.md` body plus a manifest of bundled files (so the model knows what
`read_skill_file` can fetch). Unknown name → friendly error listing available skills.

### Tool: `read_skill_file` (Level 3)

```python
def read_skill_file(
    name: Annotated[str, "Skill that owns the file."],
    path: Annotated[str, "Path relative to the skill directory, e.g. 'reference/forms.md'."],
) -> str:
    """Read a file bundled with a skill (reference docs, examples, templates)."""
```

Path-traversal guard: `(skill_dir / path).resolve()` must stay within `skill_dir.resolve()`,
else error. Missing file → friendly error.

### System-prompt contribution text

```
## Available Skills
Specialized instruction sets for specific task types. Load one before doing matching work.

- **pdf-forms**: Use when filling PDF forms or extracting fields from PDFs.
- **excel-report**: Use when creating or editing .xlsx spreadsheets.

When a task matches a skill, call `load_skill` with its name BEFORE starting, then follow
the loaded instructions exactly (they override your general approach). Load skills only
when relevant; if a skill references a file, fetch it with `read_skill_file`. If no skill
matches, proceed normally.
```

## Out of scope (future)

- Script execution (`run_skill_script`) with confirmation-strategy gating.
- Search-based discovery (`discovery="search"`) for very large skill libraries.
