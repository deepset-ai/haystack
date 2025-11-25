import time
from pathlib import Path

from deepset_cloud_sdk.workflows.sync_client.files import DeepsetCloudFile


def collect_docs_files(version: int) -> list[DeepsetCloudFile]:
    """
    Collect all HTML files from the docs and reference directories.

    Returns a list of DeepsetCloudFile objects.
    """
    repo_root = Path(__file__).parent.parent.parent
    build_dir = repo_root / "docs-website" / "build"
    # we want to exclude previous and temporarily unstable versions (2.x) and next version (next)
    exclude = ("2.", "next")

    files = []
    for section in ("docs", "reference"):
        for subfolder in (build_dir / section).iterdir():
            if subfolder.is_dir() and not any(x in subfolder.name for x in exclude):
                for html_file in subfolder.rglob("*.html"):
                    files.append(
                        DeepsetCloudFile(
                            # The build produces files like docs/agents/index.html or reference/agents-api/index.html.
                            # For file names, we want to use the parent directory name (agents.html or agents-api.html)
                            name=f"{html_file.parent.name}.html",
                            text=html_file.read_text(),
                            meta={
                                "type": "api-reference" if section == "reference" else "documentation",
                                "version": version,
                            },
                        )
                    )
    return files


if __name__ == "__main__":
    version = time.time_ns()

    dc_files = collect_docs_files(version)

    print(dc_files[:10])
    print(len(dc_files))
