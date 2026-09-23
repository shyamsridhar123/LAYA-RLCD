"""Validate notebook code/payloads; optionally execute the CPU results lesson."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
from pathlib import Path
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def validate(path: Path):
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    bundles = 0
    for cell in book.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        literals = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in ("PAYLOAD", "MANIFEST"):
                    literals[target.id] = ast.literal_eval(node.value)
        if not literals:
            continue
        assert set(literals) == {"PAYLOAD", "MANIFEST"}
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(literals["PAYLOAD"], validate=True))) as archive:
            assert len(archive.namelist()) == len(set(archive.namelist()))
            assert set(archive.namelist()) == set(literals["MANIFEST"])
            for name, digest in literals["MANIFEST"].items():
                target = (ROOT / name).resolve()
                assert target.is_relative_to(ROOT), "Unsafe archive path"
                content = archive.read(name)
                assert hashlib.sha256(content).hexdigest() == digest, name
                assert content == target.read_bytes(), f"Stale embedded file: {name}"
        bundles += 1
    assert bundles == 1, "Each lesson must have one explicit public payload"
    print(f"Validated {path.name}: code syntax, notebook format, and embedded source hashes.")
    return book


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-results", action="store_true")
    args = parser.parse_args()
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        book = validate(path)
        if args.execute_results and path.name == "03_results_walkthrough.ipynb":
            from nbclient import NotebookClient
            NotebookClient(book, timeout=180, kernel_name="python3",
                           resources={"metadata": {"path": str(ROOT)}}).execute()
            for cell in book.cells:
                cell.metadata.pop("execution", None)
            nbformat.validate(book)
            nbformat.write(book, path)
            figures = [base64.b64decode(output["data"]["image/png"])
                       for cell in book.cells for output in cell.get("outputs", [])
                       if "image/png" in output.get("data", {})]
            assert len(figures) == 2, "Expected accuracy and confusion figures"
            assets = ROOT / "docs/assets"
            assets.mkdir(parents=True, exist_ok=True)
            for name, content in zip(("accuracy.png", "decoder-confusion.png"), figures):
                (assets / name).write_bytes(content)
            print("Executed results lesson top-to-bottom and exported two figures.")


if __name__ == "__main__":
    main()
