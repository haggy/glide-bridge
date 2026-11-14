#!/usr/bin/env bash
# Download Glide MCAP bags from Google Drive into data/bags/ with rclone.
#
#   ./fetch_bags.sh setup                 # once: authorise rclone (opens a browser)
#   ./fetch_bags.sh list                  # .mcap files in the Drive folder, with sizes
#   ./fetch_bags.sh pull                  # every .mcap in the folder (skips ones you have)
#                                         # -> data/bags/<recording>/<recording>_N.mcap
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

# rosbag2 splits a recording into <recording>_0.mcap, _1.mcap, ...; a bag is the
# folder of those files. Keep that layout: data/bags/<recording>/<recording>_N.mcap,
# so the replay plays all parts of a recording in order.
recording_of() { local n="${1##*/}"; n="${n%.mcap}"; [[ "$n" =~ ^(.+)_[0-9]+$ ]] && n="${BASH_REMATCH[1]}"; echo "$n"; }

# Move loose .mcap files (older flat layout, or fresh `get` downloads) into their folders.
organize() {
    local f rec
    shopt -s nullglob
    for f in "$DEST"/*.mcap "$DEST"/.incoming/*.mcap; do
        rec="$(recording_of "$f")"
        mkdir -p "$DEST/$rec"
        mv -n "$f" "$DEST/$rec/" && echo "  $rec/$(basename "$f")"
    done
    shopt -u nullglob
}

show_bags() {
    local d
    for d in "$DEST"/*/; do
        compgen -G "$d*.mcap" >/dev/null || continue
        printf '%s  (%s, %s)\n' "${d%/}" "$(ls "$d"*.mcap | wc -l | tr -d ' ') file(s)" \
            "$(du -sh "$d" | cut -f1)"
    done
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
        organize
        pattern="*.mcap"
        [[ $# -gt 0 ]] && pattern="*$1*.mcap"
        # copyto per file into its recording folder. Never deletes local files; skips
        # files already present with the same size and time.
        rclone lsf "${REMOTE}:" "${FOLDER_ARGS[@]}" --files-only --include "$pattern" |
        while IFS= read -r name; do
            rclone copyto "${REMOTE}:$name" "$DEST/$(recording_of "$name")/$name" \
                "${FOLDER_ARGS[@]}" --progress
        done
        show_bags
        ;;
    get)
        need_rclone
        [[ $# -gt 0 ]] || die "usage: ./fetch_bags.sh get <drive URL or file id> ..."
        mkdir -p "$DEST/.incoming"
        for arg in "$@"; do
            rclone backend copyid "${REMOTE}:" "$(drive_id "$arg")" "$DEST/.incoming/" --progress
        done
        organize
        rmdir "$DEST/.incoming" 2>/dev/null || true
        show_bags
        ;;
    help|-h|--help)
        sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
        ;;
    *)
        die "unknown command '$cmd' (setup | list | pull | get | help)"
        ;;
esac
