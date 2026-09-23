"""Check Git candidates and decoded notebook bundles without printing matched values.

This is a review aid, not a certification that arbitrary material is public.
Run from any directory: python scripts/audit_publication.py
"""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
OMIT_DIRS = {".git", ".venv", ".venv-train", "node_modules", "__pycache__", ".pytest_cache",
             ".ipynb_checkpoints", "models", "runs", "recordings"}
OMIT_SUFFIXES = {".log", ".safetensors", ".pt", ".pth", ".ckpt", ".zip",
                 ".webm", ".mp4", ".pem", ".key", ".pyc"}
PUBLIC_DIRS = {".github", "benchmarks", "data", "docs", "examples", "game",
               "notebooks", "rlcd", "scripts", "tests"}
PUBLIC_ROOT_FILES = {".gitattributes", ".gitignore", "LICENSE", "NOTICE.md",
                     "README.md", "CONTRIBUTING.md", "package.json", "package-lock.json",
                     "tsconfig.json", "pytest.ini", "requirements.txt",
                     "requirements-dev.txt", "requirements-models.txt"}
PATTERNS = {
    "GitHub token": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,})\b",
    "Hub token": r"\bhf_[A-Za-z0-9]{25,}\b",
    "API secret": r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b",
    "AWS access key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "Private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "Bearer credential": r"(?i)bearer\s+[A-Za-z0-9_.-]{24,}",
    "JWT": r"\beyJ[A-Za-z0-9_-]{15,}\.eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\b",
    "Personal Windows path": r"(?i)[A-Z]:[\\/]+Users[\\/]+[^\s\\/\"']+",
    "Personal Unix path": r"/(?:Users|home)/[A-Za-z0-9_.-]+/",
    "Private notebook link": r"https://colab\.research\.google\.com/drive/[A-Za-z0-9_-]+",
    "Drive file link": r"https://drive\.google\.com/file/d/[A-Za-z0-9_-]+",
    "Credential URL": r"https?://[^\s/:'\"]+:[^\s/@'\"]+@",
    "Signed URL": r"(?i)[?&](?:sig|X-Amz-Signature|X-Goog-Signature)=[A-Za-z0-9%+/=_-]{12,}",
}
COMPILED = {name: re.compile(pattern) for name, pattern in PATTERNS.items()}


def ignored(path: Path) -> bool:
    return (any(part in OMIT_DIRS for part in path.parts)
            or path.parts[:2] == ("game", "dist")
            or path.suffix.lower() in OMIT_SUFFIXES
            or path.name.startswith(".env")
            or path.name in {".DS_Store", "Thumbs.db"}
            or ("executed" in path.name and path.suffix == ".ipynb"))


def candidates() -> list[Path]:
    if (ROOT / ".git").exists():
        result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                                cwd=ROOT, check=True, capture_output=True)
        # Include tracked files even if they have since become ignored.
        return sorted({Path(x.decode("utf-8")) for x in result.stdout.split(b"\0") if x})
    paths = []
    for directory, folders, files in os.walk(ROOT):
        folders[:] = [name for name in folders if not ignored((Path(directory) / name).relative_to(ROOT))]
        for name in files:
            relative = (Path(directory) / name).relative_to(ROOT)
            if not ignored(relative):
                paths.append(relative)
    return sorted(paths)


def scan_text(text: str) -> set[str]:
    return {name for name, pattern in COMPILED.items() if pattern.search(text)}


def png_text(content: bytes) -> list[str]:
    if not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Invalid PNG")
    texts, offset = [], 8
    while offset + 12 <= len(content):
        length = struct.unpack(">I", content[offset:offset + 4])[0]
        kind = content[offset + 4:offset + 8]
        payload = content[offset + 8:offset + 8 + length]
        if len(payload) != length:
            raise ValueError("Truncated PNG")
        if kind == b"tEXt":
            texts.append(payload.decode("latin-1"))
        elif kind == b"zTXt":
            key, rest = payload.split(b"\0", 1)
            texts.extend([key.decode("latin-1"), zlib.decompress(rest[1:]).decode("latin-1")])
        elif kind in {b"iTXt", b"eXIf"}:
            # Rich metadata is unnecessary in the public charts/screenshots.
            raise ValueError("Review/remove rich PNG metadata")
        offset += length + 12
        if kind == b"IEND":
            return texts
    raise ValueError("Missing PNG end marker")


def notebook_parts(content: bytes):
    notebook = json.loads(content)
    yield "metadata", json.dumps(notebook.get("metadata", {})).encode()
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        yield f"cell {index} source", source.encode()
        yield f"cell {index} metadata", json.dumps(cell.get("metadata", {})).encode()
        if cell["cell_type"] == "code":
            literals = {}
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                    if isinstance(target, ast.Name) and target.id in {"PAYLOAD", "MANIFEST"}:
                        literals[target.id] = ast.literal_eval(node.value)
            if literals:
                if set(literals) != {"PAYLOAD", "MANIFEST"}:
                    raise ValueError("Incomplete notebook bundle")
                with zipfile.ZipFile(io.BytesIO(base64.b64decode(literals["PAYLOAD"], validate=True))) as bundle:
                    if (len(bundle.namelist()) != len(set(bundle.namelist()))
                            or set(bundle.namelist()) != set(literals["MANIFEST"])):
                        raise ValueError("Notebook bundle allowlist mismatch")
                    if sum(info.file_size for info in bundle.infolist()) > 20_000_000:
                        raise ValueError("Oversized notebook bundle")
                    for name in bundle.namelist():
                        member = PurePosixPath(name)
                        if member.is_absolute() or ".." in member.parts or "\\" in name:
                            raise ValueError("Unsafe notebook member")
                        data = bundle.read(name)
                        target = (ROOT / name).resolve()
                        if not target.is_relative_to(ROOT) or target.is_symlink():
                            raise ValueError("Notebook member outside repository")
                        if hashlib.sha256(data).hexdigest() != literals["MANIFEST"][name]:
                            raise ValueError("Notebook bundle hash mismatch")
                        if data != target.read_bytes():
                            raise ValueError("Notebook bundle differs from source")
                        yield f"cell {index} bundle/{name}", data
        for number, output in enumerate(cell.get("outputs", [])):
            prefix = f"cell {index} output {number}"
            for key in ("text", "ename", "evalue", "traceback", "metadata"):
                if key in output:
                    yield prefix, json.dumps(output[key]).encode()
            for mime, value in output.get("data", {}).items():
                value = "".join(value) if isinstance(value, list) else value
                if mime == "image/png":
                    yield prefix + ".png", base64.b64decode(value)
                else:
                    yield prefix, (value if isinstance(value, str) else json.dumps(value)).encode()


def inspect_bytes(content: bytes, png: bool = False) -> set[str]:
    if png:
        return set().union(*(scan_text(text) for text in png_text(content)))
    return scan_text(content.decode("utf-8"))


def main():
    issues = []
    paths = candidates()
    for relative in paths:
        name = relative.as_posix()
        path = ROOT / relative
        if ignored(relative) or (len(relative.parts) == 1 and name not in PUBLIC_ROOT_FILES) or (
                len(relative.parts) > 1 and relative.parts[0] not in PUBLIC_DIRS):
            issues.append((name, "Unexpected publication path"))
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            issues.append((name, "Symlink or outside repository"))
            continue
        try:
            content = path.read_bytes()
            if len(content) > 5_000_000:
                raise ValueError("Oversized public file")
            parts = notebook_parts(content) if path.suffix == ".ipynb" else [("", content)]
            for part, data in parts:
                location = name + (f" [{part}]" if part else "")
                for rule in inspect_bytes(data, path.suffix == ".png" or part.endswith(".png")):
                    issues.append((location, rule))
        except (ValueError, OSError, KeyError, SyntaxError, zipfile.BadZipFile) as error:
            # Error messages from parsers may contain input, so do not echo them.
            issues.append((name, f"Could not fully inspect ({type(error).__name__})"))
    lock = json.loads((ROOT / "package-lock.json").read_text(encoding="utf-8"))
    for package, value in lock.get("packages", {}).items():
        resolved = value.get("resolved")
        if resolved and not resolved.startswith("https://registry.npmjs.org/"):
            issues.append(("package-lock.json", "Non-public npm registry URL"))
    if issues:
        for name, rule in sorted(set(issues)):
            print(f"REVIEW {name}: {rule}")
        raise SystemExit(1)
    print(f"Reviewed {len(paths)} publication candidates, decoded notebook bundles/outputs, "
          "PNG metadata, and npm registry URLs: no configured findings.")
    print("Review remains necessary for confidential information these generic patterns cannot identify.")


if __name__ == "__main__":
    main()
