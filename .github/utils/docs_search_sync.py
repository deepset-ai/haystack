import os
from pathlib import Path


def transform_filename(filepath: Path) -> str:
    """
    Transform a file path to the required format:

    - Use parent directory name + .html instead of index.html
    - For other HTML files, use parent_dir_filename.html
    """
    base_dir = Path(__file__).parent
    # print(base_dir)
    # Get relative path from base directory
    rel_path = filepath.relative_to(base_dir)

    # Get parent directory name
    parent_dir = rel_path.parent.name if rel_path.parent.name else "root"

    # Transform index.html to parent_dir.html
    if filepath.name == "index.html":
        transformed = f"{parent_dir}.html"
    else:
        # For other HTML files, use parent_dir_filename.html
        transformed = f"{parent_dir}_{filepath.name}"

    return transformed


if __name__ == "__main__":
    files = []

    dirs = ["../docs-website/build/docs", "../docs-website/build/reference"]
    for dir_path in dirs:
        # we want to exclude previous versions (2.x) and next version (next)
        # also excluding unstable version when manually running the script
        substrings_to_exclude = ["2.", "next", "unstable"]
        subfolders = [
            f.path
            for f in os.scandir(dir_path)
            if f.is_dir() and not any(substring in f.path for substring in substrings_to_exclude)
        ]

        for subfolder in subfolders:
            # print(subfolder)
            for html_file in Path(subfolder).rglob("*.html"):
                doc_type = "documentation"
                if "/reference/" in html_file.as_posix():
                    doc_type = "api-reference"

                files.append((transform_filename(html_file), html_file, doc_type))

    print(files[:10])
    print(len(files))
