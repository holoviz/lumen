"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src = Path("lumen")

for path in sorted(src.rglob("*.py")):
    # Skip private modules and test files
    if any(part.startswith("_") for part in path.parts):
        continue
    if "tests" in path.parts:
        continue

    module_path = path.with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    # Skip the top-level "lumen" in navigation by using parts[1:]
    parts = tuple(module_path.parts)
    if len(parts) > 1:
        nav[parts[1:]] = doc_path.as_posix()  # Remove "lumen" from nav
    else:
        nav[parts] = doc_path.as_posix()  # Keep single-level modules

    with mkdocs_gen_files.open(full_doc_path, "w") as f:
        identifier = ".".join(parts)
        f.write(f"::: {identifier}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
