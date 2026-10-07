# SPDX-FileCopyrightText: 2026 Sankalp Gilda
#
# SPDX-License-Identifier: Apache-2.0

"""Qualify a Haystack host publication gate with installed isolated readers."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

_OBSERVER_PIN = "5d9feedcfffcf441380e11e9f25a5db44d274068"
_PROFILE = Path("interop/haystack-native-2026-10-03")


def _encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_bytes(source: Path, member: dict) -> tuple[Path, bytes]:
    rel = Path(member["path"])
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("source path escapes selection")
    path = source / rel
    if path.is_symlink() or any(p.is_symlink() for p in path.parents if p.is_relative_to(source)):
        raise ValueError("source must not be a symlink")
    body = path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    if blob != member["sha"]:
        raise ValueError(f"source blob differs: {rel}")
    return rel, body


def _check_source(source: Path, selection: Path) -> dict:
    data = json.loads(selection.read_text())
    if data["repo"] != "probityai/agent-evidence-observer" or data["commit"] != _OBSERVER_PIN:
        raise ValueError("unexpected source selection")
    for member in data["files"]:
        _selected_bytes(source, member)
    return data


def _stage_selected_source(source: Path, destination: Path, data: dict) -> None:
    # Recheck the bytes that will enter the build; ignore every unselected file.
    selected = [_selected_bytes(source, member) for member in data["files"]]
    destination.mkdir()
    for rel, body in selected:
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)


def _clean_environment(out: Path) -> dict[str, str]:
    env = dict(os.environ)
    for name in list(env):
        if (
            name == "VIRTUAL_ENV"
            or any(word in name for word in ("TOKEN", "SECRET", "API_KEY", "PASSWORD"))
            or name.startswith(("PYTHON", "PIP_", "HATCH_"))
        ):
            env.pop(name)
    temporary = out / "temporary"
    temporary.mkdir()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        HAYSTACK_TELEMETRY_ENABLED="false",
        TMPDIR=str(temporary),
        SOURCE_DATE_EPOCH="946684800",
        UV_CACHE_DIR=str(out / "uv-cache"),
        UV_PYTHON_INSTALL_DIR=str(out / "uv-python"),
        UV_LINK_MODE="copy",
        HATCH_DATA_DIR=str(out / "hatch-data"),
        HATCH_CACHE_DIR=str(out / "hatch-cache"),
    )
    return env


class _Commands:
    def __init__(self, out: Path, env: dict[str, str]):
        self.out, self.env, self.rows = out, env, []

    def run(
        self, name: str, command: list[str], *, expected: int | None = 0, timeout: int = 600, cwd: Path | None = None
    ):
        start = time.monotonic()
        row = {
            "name": name,
            "command": command,
            "cwd": str(cwd or self.out),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        process = None
        with (self.out / (name + ".stdout")).open("wb") as stdout, (self.out / (name + ".stderr")).open("wb") as stderr:
            try:
                process = subprocess.Popen(
                    command, cwd=cwd or self.out, env=self.env, stdout=stdout, stderr=stderr, start_new_session=True
                )
                row["process_group"] = process.pid
                row["exit_code"] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                row["exit_code"] = None
                row["timeout"] = True
            except OSError as error:
                row.update(exit_code=None, launch_error=type(error).__name__, errno=error.errno)
            except KeyboardInterrupt:
                row.update(exit_code=None, cancelled=True)
            finally:
                if process is not None:
                    try:
                        row["live_group_members_before_cleanup"] = _live_group_members(process.pid)
                        row["process_group_clear"] = _stop_process_group(process)
                    except (OSError, subprocess.TimeoutExpired) as error:
                        row.update(process_group_clear=False, cleanup_error=type(error).__name__)
                    row["process_returncode"] = process.returncode
        row["elapsed_seconds"] = time.monotonic() - start
        self.rows.append(row)
        (self.out / "commands.json").write_bytes(_encoded(self.rows))
        print(name, row["exit_code"], flush=True)
        if row.get("cancelled"):
            raise KeyboardInterrupt
        if (
            row.get("timeout")
            or row.get("launch_error")
            or row.get("process_group_clear") is False
            or expected is not None
            and row["exit_code"] != expected
        ):
            raise RuntimeError(f"{name} failed; retained stdout/stderr and process receipt")
        return row


def _live_group_members(group: int) -> list[int]:
    members = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == group and fields[0] != "Z":
                    members.append(int(entry.name))
            except (FileNotFoundError, ProcessLookupError):
                pass
    return members


def _stop_process_group(process: subprocess.Popen) -> bool:
    for action in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, action)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        deadline = time.monotonic() + 2
        while _live_group_members(process.pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        if not _live_group_members(process.pid):
            break
    process.wait(timeout=2)
    return not _live_group_members(process.pid)


def _reader_command(python: Path, packet: Path, policy: Path, publication: Path | None = None):
    command = [
        str(python),
        "-I",
        "-B",
        "-m",
        "probity_haystack.reader",
        str(packet),
        "--host-policy",
        str(policy),
        "--policy-sha256",
        _sha(policy),
    ]
    return command + (["--publish", str(publication)] if publication else [])


def _framework_selection(source: Path, captured: Path | None) -> dict:
    if captured:
        data = json.loads(captured.read_text())
    else:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        tree = subprocess.check_output(
            ["git", "ls-tree", "-rz", "HEAD", "--", "haystack", "pyproject.toml", "VERSION.txt", "README.md"],
            cwd=source,
        )
        files = []
        for row in tree.split(b"\0"):
            if row:
                fields, path = row.split(b"\t", 1)
                mode, kind, blob = fields.decode().split()
                if kind != "blob" or mode not in ("100644", "100755"):
                    raise ValueError("unsupported host source member")
                files.append({"path": path.decode(), "sha": blob})
        data = {"repo": "deepset-ai/haystack", "commit": commit, "files": files}
    if data["repo"] != "deepset-ai/haystack" or len(data["commit"]) != 40 or not data["files"]:
        raise ValueError("unexpected host source selection")
    if any(char not in "0123456789abcdef" for char in data["commit"]):
        raise ValueError("invalid host source commit")
    return data


def _derive_host_reference(source_copy: Path, framework: dict, version: str, out: Path) -> dict:
    profile = source_copy / _PROFILE
    contract = profile / "probity_haystack/contract.py"
    original = contract.read_bytes()
    declarations = {
        "SDK_VERSION": ('SDK_VERSION = "3.3.0"\n', version),
        "SDK_SOURCE": ('SDK_SOURCE = "daa2d1ffacd083dcb1fc9a455adf541360a9e09c"\n', framework["commit"]),
    }
    lines = original.decode().splitlines(keepends=True)
    for name, (expected, value) in declarations.items():
        if lines.count(expected) != 1:
            raise ValueError("original SDK identity declarations differ")
        lines[lines.index(expected)] = name + " = " + json.dumps(value) + "\n"
    derived = "".join(lines).encode()
    metadata = profile / "pyproject.toml"
    metadata_original = metadata.read_bytes()
    text = metadata_original.decode()
    expected_version = 'version = "0.0.1"\n'
    expected_extra = 'producer = ["haystack-ai==3.3.0"]\n'
    if text.count(expected_version) != 1 or text.count(expected_extra) != 1:
        raise ValueError("original reference metadata differs")
    local_version = "0.0.1+host." + framework["commit"][:12]
    text = text.replace(expected_version, "version = " + json.dumps(local_version) + "\n", 1)
    text = text.replace(expected_extra, "producer = [" + json.dumps("haystack-ai==" + version) + "]\n", 1)
    text += '\n[tool.setuptools.package-data]\nprobity_haystack = ["host-reference-derivation.json"]\n'
    metadata_derived = text.encode()
    retained = out / "reference-derivation"
    retained.mkdir()
    for name, before, after in [
        ("contract.py", original, derived),
        ("pyproject.toml", metadata_original, metadata_derived),
    ]:
        (retained / ("original-" + name)).write_bytes(before)
        (retained / ("derived-" + name)).write_bytes(after)
        difference = "".join(
            difflib.unified_diff(
                before.decode().splitlines(keepends=True),
                after.decode().splitlines(keepends=True),
                fromfile="original/" + name,
                tofile="derived/" + name,
            )
        )
        (retained / (name + ".diff")).write_text(difference)
    receipt = {
        "schema": "haystack-host-reference-derivation-v1",
        "base_observer_commit": _OBSERVER_PIN,
        "host_commit": framework["commit"],
        "host_sdk_version": version,
        "derived_distribution_version": local_version,
        "runtime_selector_changes": ["SDK_VERSION", "SDK_SOURCE"],
        "original_contract_sha256": hashlib.sha256(original).hexdigest(),
        "derived_contract_sha256": hashlib.sha256(derived).hexdigest(),
        "original_metadata_sha256": hashlib.sha256(metadata_original).hexdigest(),
        "derived_metadata_sha256": hashlib.sha256(metadata_derived).hexdigest(),
        "original_tests_sha256": _sha(profile / "test_native.py"),
        "substantive_guards_changed": False,
    }
    contract.write_bytes(derived)
    metadata.write_bytes(metadata_derived)
    (retained / "receipt.json").write_bytes(_encoded(receipt))
    (profile / "probity_haystack/host-reference-derivation.json").write_bytes(_encoded(receipt))
    return receipt


def _qualify(
    source: Path,
    selection: Path,
    out: Path,
    python: str,
    framework_source: Path | None = None,
    captured_framework: Path | None = None,
) -> dict:
    out.mkdir()
    data = _check_source(source, selection)
    (out / "source-selection.json").write_bytes(_encoded(data))
    commands = _Commands(out, _clean_environment(out))
    source_copy = out / "build-source"
    _stage_selected_source(source, source_copy, data)
    producer = out / "producer/bin/python"
    reader = out / "reader/bin/python"
    commands.run("producer-venv", ["uv", "venv", str(out / "producer"), "--python", python])
    commands.run("reader-venv", ["uv", "venv", str(out / "reader"), "--python", python])
    for label, executable, lock in [
        ("producer", producer, "requirements.lock"),
        ("reader", reader, "requirements-reader.lock"),
    ]:
        commands.run(
            label + "-dependencies",
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(executable),
                "--require-hashes",
                "-r",
                str(source_copy / _PROFILE / lock),
            ],
        )
    framework = None
    derivation = None
    if framework_source:
        framework = _framework_selection(framework_source, captured_framework)
        (out / "framework-source-selection.json").write_bytes(_encoded(framework))
        framework_copy = out / "framework-build-source"
        _stage_selected_source(framework_source, framework_copy, framework)
        commands.run(
            "framework-wheel", ["hatch", "build", "-t", "wheel", str(out / "framework-wheels")], cwd=framework_copy
        )
        host_wheels = list((out / "framework-wheels").glob("*.whl"))
        if len(host_wheels) != 1:
            raise ValueError("expected one host framework wheel")
        commands.run(
            "framework-install",
            ["uv", "pip", "install", "--python", str(producer), "--no-deps", "--reinstall", str(host_wheels[0])],
        )
        commands.run("framework-dependencies-check", ["uv", "pip", "check", "--python", str(producer)])
        commands.run(
            "framework-installed-source",
            [
                str(producer),
                "-I",
                "-B",
                "-c",
                "import hashlib,importlib.metadata,json,pathlib,haystack; from packaging.version import Version; "
                "data=json.loads(pathlib.Path(__import__('sys').argv[1]).read_text()); "
                "root=pathlib.Path(haystack.__file__).parent.parent; "
                "members=[m for m in data['files'] if m['path'].startswith('haystack/')]; "
                "bodies=[(m,(root/m['path']).read_bytes()) for m in members]; "
                "assert all(hashlib.sha1(b'blob '+str(len(b)).encode()+b'\\0'+b).hexdigest()==m['sha'] "
                "for m,b in bodies),'installed host source differs'; "
                "assert Version(importlib.metadata.version('haystack-ai'))==Version("
                "pathlib.Path(__import__('sys').argv[2]).read_text().strip()),'host version differs'; "
                "print(json.dumps({'installed_files':len(members),'version':importlib.metadata.version('haystack-ai')}))",
                str(out / "framework-source-selection.json"),
                str(framework_copy / "VERSION.txt"),
            ],
        )
        _check_source(source_copy, selection)
        installed = json.loads((out / "framework-installed-source.stdout").read_text())
        derivation = _derive_host_reference(source_copy, framework, installed["version"], out)
    wheels = out / "wheels"
    for label, directory in [("observer", source_copy), ("haystack-profile", source_copy / _PROFILE)]:
        commands.run(
            label + "-wheel",
            [
                str(producer),
                "-I",
                "-B",
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(wheels),
                str(directory),
            ],
        )
    selected_wheels = sorted(str(p) for p in wheels.glob("*.whl"))
    if len(selected_wheels) != 2:
        raise ValueError("expected exactly two wheels")
    for label, executable in [("producer", producer), ("reader", reader)]:
        commands.run(
            label + "-install", ["uv", "pip", "install", "--python", str(executable), "--no-deps", *selected_wheels]
        )
        commands.run(label + "-freeze", ["uv", "pip", "freeze", "--python", str(executable)])
    if derivation:
        commands.run(
            "derived-reference-receipt",
            [
                str(reader),
                "-I",
                "-B",
                "-c",
                "import importlib.metadata,json,pathlib; import probity_haystack.contract as contract; "
                "expected=json.loads(pathlib.Path(__import__('sys').argv[1]).read_text()); "
                "actual=json.loads(pathlib.Path(contract.__file__)."
                "with_name('host-reference-derivation.json').read_text()); "
                "assert actual==expected,'installed derivation receipt differs'; "
                "assert contract.SDK_VERSION==expected['host_sdk_version'] and "
                "contract.SDK_SOURCE==expected['host_commit'],'derived selectors differ'; "
                "assert importlib.metadata.version('probity-haystack-reference')=="
                "expected['derived_distribution_version'], "
                "'derived distribution version differs'; print(json.dumps(actual))",
                str(out / "reference-derivation/receipt.json"),
            ],
        )
    commands.run(
        "reader-framework-absence",
        [str(reader), "-I", "-c", "import importlib.util; assert importlib.util.find_spec('haystack') is None"],
    )
    commands.env["HAYSTACK_READER_PYTHON"] = str(reader)
    commands.run(
        "upstream-native-controls",
        [
            str(producer),
            "-I",
            "-B",
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            str(source_copy / _PROFILE / "test_native.py"),
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp=" + str(out / "native-test-temporary"),
            "--junitxml=" + str(out / "native-tests.xml"),
        ],
    )
    packet = out / "packet"
    commands.run("fresh-native-producer", [str(producer), "-I", "-B", "-m", "probity_haystack.producer", str(packet)])
    # Keep the selected policy outside the candidate packet; custody remains local.
    policy = out / "host-selected-policy.json"
    shutil.copyfile(packet / "host-policy.json", policy)
    commands.run("reader-first", _reader_command(reader, packet, policy))
    commands.run("reader-second", _reader_command(reader, packet, policy))
    if (out / "reader-first.stdout").read_bytes() != (out / "reader-second.stdout").read_bytes():
        raise ValueError("repeated installed reader decisions differ")
    publication = out / "publication"
    commands.run("publish-original", _reader_command(reader, packet, policy, publication))
    original_publication = _sha(publication / "published.json")
    # A candidate rewrites both its tool trace and digest. Signed effects still bind original bytes.
    altered = out / "changed-input-packet"
    shutil.copytree(packet, altered)
    native_path = altered / "permit/native.json"
    native = json.loads(native_path.read_text())
    native["toolCalls"][0]["content"] = "changed after signed permission"
    native_path.write_bytes(_encoded(native))
    changed_policy_data = json.loads(policy.read_text())
    changed_policy_data["cases"]["permit"]["artifacts"]["native.json"] = _sha(native_path)
    changed_policy = out / "host-selected-changed-policy.json"
    changed_policy.write_bytes(_encoded(changed_policy_data))
    changed_publication = out / "changed-input-publication"
    result = commands.run(
        "refuse-rehashed-changed-input",
        _reader_command(reader, altered, changed_policy, changed_publication),
        expected=None,
    )
    if result["exit_code"] == 0 or changed_publication.exists():
        raise ValueError("changed invocation admitted or refusal wrote a publication")
    if b"invocation bytes differ" not in (out / "refuse-rehashed-changed-input.stderr").read_bytes():
        raise ValueError("changed-input check did not reach the expected semantic refusal")
    commands.run(
        "refuse-original-policy-change",
        _reader_command(reader, altered, policy, out / "unselected-publication"),
        expected=None,
    )
    if commands.rows[-1]["exit_code"] == 0 or (out / "unselected-publication").exists():
        raise ValueError("changed candidate passed original host selection")
    if _sha(publication / "published.json") != original_publication:
        raise ValueError("refused candidate changed the original publication")
    report = json.loads((out / "reader-first.stdout").read_text())
    released = json.loads((publication / "published.json").read_text())
    if len(report["records"]) != 7 or [x["case"] for x in released["records"]] != ["permit"]:
        raise ValueError("unexpected publication population")
    completion = {
        "schema": "haystack-host-gate-qualification-v1",
        "source_commit": _OBSERVER_PIN,
        "source_selection_sha256": _sha(selection),
        "native_records": 7,
        "released_cases": ["permit"],
        "changed_input_semantic_refused": True,
        "changed_input_pin_refused": True,
        "publication_unchanged": True,
        "repeated_reader_bytes_equal": True,
        "reader_has_haystack": False,
        "model_inference_calls": 0,
        "independent_custody": False,
        "runtime_context": "unverified; see external CI or operator receipt",
        "reader_report": report,
        "wheels": [{"name": p.name, "sha256": _sha(p)} for p in sorted(wheels.glob("*.whl"))],
    }
    if framework:
        completion["host_framework"] = {
            "repo": framework["repo"],
            "commit": framework["commit"],
            "selected_source_files": len(framework["files"]),
            "source_selection_sha256": _sha(out / "framework-source-selection.json"),
            "wheels": [{"name": p.name, "sha256": _sha(p)} for p in host_wheels],
        }
        completion["reference_derivation"] = derivation
    (out / "COMPLETE.json").write_bytes(_encoded(completion))
    return completion


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default="python3.13")
    parser.add_argument("--framework-source", type=Path)
    parser.add_argument("--framework-selection", type=Path)
    args = parser.parse_args()
    if args._framework_selection and not args.framework_source:
        parser.error("--framework-selection needs --framework-source")
    print(
        json.dumps(
            _qualify(
                args.source.resolve(),
                args.selection.resolve(),
                args.output.resolve(),
                args.python,
                args.framework_source.resolve() if args.framework_source else None,
                args._framework_selection.resolve() if args._framework_selection else None,
            )
        )
    )


if __name__ == "__main__":
    _main()
