#!/usr/bin/env bash
# Export the glide_ros environment for VS Code's debugger.
#
# VS Code can't `source env/activate_ros.sh`, so this does it once and writes:
#   .vscode/ros.env      every variable the activation sets or changes (dotenv format,
#                        loaded via "envFile" in .vscode/launch.json)
#   .vscode/ros-python   symlink to the glide_ros interpreter ("python" in launch.json)
# Run by the "glide_ros env" task (.vscode/tasks.json) before each ROS debug session,
# so it stays correct after env updates. Both outputs are git-ignored.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .vscode

before="$(mktemp)"
trap 'rm -f "$before"' EXIT
env -0 > "$before"
source env/activate_ros.sh > /dev/null

python - "$before" <<'EOF'
import os, sys
from pathlib import Path

before = dict(kv.split("=", 1) for kv in Path(sys.argv[1]).read_text().split("\0") if "=" in kv)
skip = {"PWD", "OLDPWD", "SHLVL", "_"}
lines = [f'{k}="{v}"' for k, v in sorted(os.environ.items())
         if k not in skip and before.get(k) != v and "\n" not in v and '"' not in v]
out = Path(".vscode/ros.env")
tmp = out.with_suffix(f".env.{os.getpid()}")   # write-then-rename: configs may run this concurrently
tmp.write_text("\n".join(lines) + "\n")
tmp.replace(out)

link = Path(".vscode/ros-python")
tmp_link = link.with_name(f"ros-python.{os.getpid()}")
tmp_link.symlink_to(sys.executable)
tmp_link.replace(link)
print(f"wrote {out} ({len(lines)} vars), {link} -> {sys.executable}")
EOF
