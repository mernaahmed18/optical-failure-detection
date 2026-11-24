"""Build and execute .ipynb notebooks from nb_src/*.py scripts with Run All."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "nb_src"


def py_to_nb(py_path: Path) -> nbformat.NotebookNode:
    """Parse # %% and # %% [markdown] blocks into an nbformat v4 notebook."""
    text = py_path.read_text(encoding="utf-8")
    cells = []
    cur_type = None
    cur_lines: list[str] = []

    def flush():
        if cur_type is None:
            return
        body = "\n".join(cur_lines).strip("\n")
        if cur_type == "markdown":
            md_lines = [l[2:] if l.startswith("# ") else ("" if l == "#" else l) for l in body.splitlines()]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(md_lines)))
        else:
            if body.strip():
                cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.strip() == "# %% [markdown]":
            flush()
            cur_type = "markdown"
            cur_lines = []
        elif line.strip() == "# %%":
            flush()
            cur_type = "code"
            cur_lines = []
        else:
            if cur_type is None:
                cur_type = "code"
            cur_lines.append(line)
    flush()

    nb = nbformat.v4.new_notebook(cells=cells)
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def run_notebook(name: str, timeout: int = 900) -> None:
    """Convert nb_src/<name>.py -> <name>.ipynb, execute all cells, and save."""
    py_path = SRC_DIR / f"{name}.py"
    nb_path = ROOT / f"{name}.ipynb"
    nb = py_to_nb(py_path)
    # Use inline matplotlib backend so plots render directly inside notebook cell outputs
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.write(nb, str(nb_path))
    print(f"OK: {nb_path.name} ({len(nb.cells)} cells executed)")


if __name__ == "__main__":
    targets = sys.argv[1:]
    if not targets:
        targets = sorted(p.stem for p in SRC_DIR.glob("*.py"))
    for t in targets:
        run_notebook(t)
