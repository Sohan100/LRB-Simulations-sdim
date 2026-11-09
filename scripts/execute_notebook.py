"""Execute a notebook in place with the lrb-sdim140 kernel from the repo root.

Usage: PYTHONNOUSERSITE=1 .venv-sdim140/bin/python scripts/execute_notebook.py NOTEBOOK
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1]).resolve()
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=None, kernel_name="lrb-sdim140",
                        resources={"metadata": {"path": str(path.parent)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print(f"executed {path}")
