import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
if (
    root.parts[-3:] != ("computing_math", "cf1305e_test_generation", "base")
    or root.is_symlink()
):
    raise RuntimeError("Invalid task root")
expected = json.loads(Path(sys.argv[2]).read_text())
for name, digest in expected.items():
    path = root / name
    if (
        path.is_symlink()
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != digest
    ):
        raise RuntimeError(f"Initial task material missing or corrupt: {name}")
for command in (["/usr/bin/g++", "--version"], ["/usr/bin/python3", "--version"]):
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
if os.geteuid() != 0:
    raise RuntimeError("Initialization requires sudo for evaluator isolation")
for name in ("output", "reference"):
    path = root / name
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
out = root / "output"
out.mkdir(mode=0o755)
owner = (root / "input").stat()
uid = owner.st_uid if owner.st_uid else 1000
gid = owner.st_gid if owner.st_gid else 1000
os.chown(out, uid, gid)
shutil.copyfile(root / "input/gen.cpp", out / "gen.cpp")
os.chown(out / "gen.cpp", uid, gid)
if (root / "reference").exists():
    raise RuntimeError("Reference cleanup failed")
print("dispatch 17 staged; private reference absent")
