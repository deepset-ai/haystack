#!/usr/bin/env python3
"""HTTPXodus 侦察脚本：验证种子清单中每个仓库的 httpx 依赖现状。

对每个仓库：
1. 拉取基本元数据（stars / 是否归档 / 最近 push）
2. 检查依赖清单（pyproject.toml / requirements.txt / setup.py / setup.cfg），
   判断是否仍依赖 httpx、是否已迁移 httpx2
3. 搜索已有 httpx2 相关 issue/PR（查重）
4. 生成 targets/board.md 和 targets/recon.json

用法：python3 scripts/recon.py [seed文件路径]
需要：已认证的 gh CLI。
"""
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "targets" / "seed.txt"
OUT_JSON = ROOT / "targets" / "recon.json"
OUT_BOARD = ROOT / "targets" / "board.md"

MANIFESTS = ["pyproject.toml", "requirements.txt", "setup.py", "setup.cfg"]


def gh_api(path: str, *, retries: int = 2) -> dict | list | None:
    for attempt in range(retries + 1):
        r = subprocess.run(
            ["gh", "api", path],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0:
            try:
                return json.loads(r.stdout)
            except json.JSONDecodeError:
                return None
        # 404/451 等确定性错误不重试，避免浪费 6s+
        if "HTTP 4" in r.stderr:
            return None
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    return None


def fetch_manifest(repo: str, name: str) -> str | None:
    data = gh_api(f"repos/{repo}/contents/{name}")
    if not isinstance(data, dict) or data.get("encoding") != "base64":
        return None
    try:
        return base64.b64decode(data["content"]).decode("utf-8", "replace")
    except Exception:
        return None


def detect_httpx_status(repo: str) -> tuple[str, str]:
    """返回 (status, evidence)。status: uses-httpx / migrated / dual / no-manifest-hit / unknown"""
    hits_httpx, hits_httpx2, files_found = [], [], []
    for name in MANIFESTS:
        content = fetch_manifest(repo, name)
        if content is None:
            continue
        files_found.append(name)
        for i, line in enumerate(content.splitlines(), 1):
            low = line.lower()
            if "httpx2" in low:
                hits_httpx2.append(f"{name}:{i}")
            elif "httpx" in low:  # 注意顺序：先判 httpx2
                hits_httpx.append(f"{name}:{i}")
    if hits_httpx2 and hits_httpx:
        return "dual", f"httpx2@{','.join(hits_httpx2)}; httpx@{','.join(hits_httpx)}"
    if hits_httpx2:
        return "migrated", ",".join(hits_httpx2)
    if hits_httpx:
        return "uses-httpx", ",".join(hits_httpx)
    # 根目录清单未命中 → 代码搜索兜底（monorepo 依赖常藏在子目录）
    found = search_code_manifests(repo)
    if found is not None:
        return found
    if files_found:
        return "no-manifest-hit", ",".join(files_found)
    return "unknown", "no manifests found"


def search_code_manifests(repo: str) -> tuple[str, str] | None:
    """用代码搜索找子目录里的 pyproject.toml 中的 httpx/httpx2 引用。"""
    data = gh_api(f"search/code?q=repo:{repo}+httpx+filename:pyproject.toml&per_page=3")
    time.sleep(2.2)  # code search 限流 30/min
    if not isinstance(data, dict):
        return None
    paths = [it["path"] for it in data.get("items", [])]
    if not paths:
        return None
    # 找到候选文件后逐个读内容确认是 httpx 还是 httpx2
    hits_httpx, hits_httpx2 = [], []
    for p in paths[:3]:
        content = fetch_manifest(repo, p)
        if content is None:
            continue
        for i, line in enumerate(content.splitlines(), 1):
            low = line.lower()
            if "httpx2" in low:
                hits_httpx2.append(f"{p}:{i}")
            elif "httpx" in low:
                hits_httpx.append(f"{p}:{i}")
    if hits_httpx2 and hits_httpx:
        return "dual", f"httpx2@{','.join(hits_httpx2)}; httpx@{','.join(hits_httpx)}"
    if hits_httpx2:
        return "migrated", ",".join(hits_httpx2)
    if hits_httpx:
        return "uses-httpx", ",".join(hits_httpx)
    return None


def search_existing(repo: str) -> list[str]:
    data = gh_api(f"search/issues?q=repo:{repo}+httpx2&per_page=5")
    time.sleep(2.2)  # search API 限流 30/min
    if not isinstance(data, dict):
        return []
    out = []
    for item in data.get("items", []):
        kind = "PR" if "pull_request" in item else "issue"
        out.append(f"{kind}#{item['number']}({item['state']})")
    return out


def main() -> None:
    repos = [
        ln.strip()
        for ln in SEED.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    results = []
    for idx, repo in enumerate(repos, 1):
        meta = gh_api(f"repos/{repo}")
        if not isinstance(meta, dict):
            print(f"[{idx}/{len(repos)}] {repo}: 仓库不存在或无权访问", flush=True)
            results.append({"repo": repo, "error": "not found"})
            continue
        status, evidence = detect_httpx_status(repo)
        existing = search_existing(repo) if status in ("uses-httpx", "dual", "no-manifest-hit") else []
        rec = {
            "repo": repo,
            "stars": meta.get("stargazers_count", 0),
            "archived": meta.get("archived", False),
            "pushed_at": (meta.get("pushed_at") or "")[:10],
            "httpx_status": status,
            "evidence": evidence,
            "existing_httpx2_refs": existing,
        }
        results.append(rec)
        print(f"[{idx}/{len(repos)}] {repo}: ★{rec['stars']} {status} {existing}", flush=True)

    results.sort(key=lambda r: r.get("stars", 0), reverse=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    lines = [
        "# HTTPXodus 目标看板",
        "",
        f"生成时间：由 recon.py 生成，共 {len(results)} 个候选",
        "",
        "| # | 仓库 | ★ | httpx 状态 | 已有 httpx2 issue/PR | 证据 |",
        "|---|------|---|-----------|---------------------|------|",
    ]
    for i, r in enumerate(results, 1):
        if "error" in r:
            lines.append(f"| {i} | {r['repo']} | - | ❓ 仓库不可达 | | |")
            continue
        existing = ", ".join(r["existing_httpx2_refs"]) or "—"
        lines.append(
            f"| {i} | [{r['repo']}](https://github.com/{r['repo']}) | {r['stars']} "
            f"| {r['httpx_status']} | {existing} | `{r['evidence'][:60]}` |"
        )
    OUT_BOARD.write_text("\n".join(lines) + "\n")
    print(f"\n完成 → {OUT_BOARD}")


if __name__ == "__main__":
    main()
