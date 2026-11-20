#!/usr/bin/env bash
# Download models at pinned revisions (harrier-oss-v1-27b is a symlink to the already-verified local copy @ 0c0fc62).
set -u
cd ~/embed-eval
H=venv/bin/hf
dl() { echo "[$(date -u +%FT%TZ)] START $1@$2"; $H download "$1" --revision "$2" --local-dir "models/$3" > "logs/download_$3.log" 2>&1; echo "[$(date -u +%FT%TZ)] END $1 exit=$?"; }
dl nvidia/Nemotron-3-Embed-8B-BF16 d1f2f25730bbd775b99b29185134bc86653bf2d1 Nemotron-3-Embed-8B-BF16 &
dl zeroentropy/zembed-1-embedding cf13c81f3274394053d166740294f7eea4586f7a zembed-1-embedding &
dl microsoft/harrier-oss-v1-0.6b f9b9dc8d367d443f2479d27aa5d8d2850c0774ee harrier-oss-v1-0.6b &
wait
for d in models/*/; do (cd "$d" && find -L . -type f -not -path "./.cache/*" | sort | xargs sha256sum) > "env/sha256_$(basename $d).txt"; done
echo "DOWNLOADS DONE"
