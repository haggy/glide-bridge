#!/usr/bin/env bash
# Download Glide MCAP bags from Google Drive into data/bags/ with rclone.
#
#   ./fetch_bags.sh setup                 # once: authorise rclone (opens a browser)
#   ./fetch_bags.sh list                  # .mcap files in the Drive folder, with sizes
#   ./fetch_bags.sh pull                  # every .mcap in the folder (skips ones you have)
#   ./fetch_bags.sh pull alpha_033_2026-11-12   # only names containing this text
#   ./fetch_bags.sh get <drive URL or file id> ...   # specific files anywhere in Drive
#
# Folder: the project's Drive "Data" folder by default; override with
# GLIDE_DRIVE_FOLDER=<folder id or URL>. Output: GLIDE_BAG_DIR (default data/bags).
# Access is read-only (rclone scope drive.readonly), with your own Google login.
set -euo pipefail
cd "$(dirname "$0")"

REMOTE="${GLIDE_RCLONE_REMOTE:-glide_gdrive}"
FOLDER="${GLIDE_DRIVE_FOLDER:-1TNXE_c0k8CZVG0gO_6XvGgu1yogj4COo}"   # Drive: .../Data
DEST="${GLIDE_BAG_DIR:-data/bags}"

die() { echo "fetch_bags: $*" >&2; exit 1; }

# Accept a bare id or any Drive URL form (/file/d/<id>/, /folders/<id>, ?id=<id>).
drive_id() {
    local s="$1"
    if [[ "$s" =~ /d/([A-Za-z0-9_-]{20,}) ]] || [[ "$s" =~ /folders/([A-Za-z0-9_-]{20,}) ]] \
        || [[ "$s" =~ [?\&]id=([A-Za-z0-9_-]{20,}) ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$s" =~ ^[A-Za-z0-9_-]{20,}$ ]]; then
        echo "$s"
    else
        die "not a Drive file id or URL: $s"
    fi
}

need_rclone() {
    command -v rclone >/dev/null || die "rclone not found; install it to ~/.local/bin (see README)"
    rclone listremotes | grep -qx "${REMOTE}:" \
        || die "no rclone remote '${REMOTE}'; run: ./fetch_bags.sh setup"
}

FOLDER_ID="$(drive_id "$FOLDER")"
FOLDER_ARGS=(--drive-root-folder-id "$FOLDER_ID")

cmd="${1:-help}"; shift || true
case "$cmd" in
    setup)
        command -v rclone >/dev/null || die "rclone not found"
        if rclone listremotes | grep -qx "${REMOTE}:"; then
            echo "remote '${REMOTE}' exists; re-authorising"
            rclone config reconnect "${REMOTE}:"
        else
            echo "A browser window will open; sign in with the account that owns the bags."
            rclone config create "$REMOTE" drive scope=drive.readonly
        fi
        rclone about "${REMOTE}:" >/dev/null && echo "ok: '${REMOTE}' can read your Drive"
        ;;
    list)
        need_rclone
        rclone lsl "${REMOTE}:" "${FOLDER_ARGS[@]}" --include "*.mcap" \
            | awk '{printf "%8.1f MB  %s %s  %s\n", $1/1e6, $2, substr($3,1,8), $4}'
        ;;
    pull)
        need_rclone
        mkdir -p "$DEST"
        filter=(--include "*.mcap")
        [[ $# -gt 0 ]] && filter=(--include "*$1*.mcap")
        # copy (not sync): never deletes local files; skips files already present.
        rclone copy "${REMOTE}:" "$DEST" "${FOLDER_ARGS[@]}" "${filter[@]}" \
            --progress --transfers 2
        ls -lh "$DEST"/*.mcap 2>/dev/null || true
        ;;
    get)
        need_rclone
        [[ $# -gt 0 ]] || die "usage: ./fetch_bags.sh get <drive URL or file id> ..."
        mkdir -p "$DEST"
        for arg in "$@"; do
            rclone backend copyid "${REMOTE}:" "$(drive_id "$arg")" "$DEST/" --progress
        done
        ls -lh "$DEST"
        ;;
    help|-h|--help)
        sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
        ;;
    *)
        die "unknown command '$cmd' (setup | list | pull | get | help)"
        ;;
esac
