#!/usr/bin/env bash
# Wrap the raw VHS recording with a title and end card.
#
#   bash scripts/finish-video.sh
#
# The middle of the video is untouched: it is the real terminal session with
# real output. Only the bookends are generated, and they only state results
# that are already in bench-results/.
set -euo pipefail

cd "$(dirname "$0")/.."

RAW="${RAW:-demo-raw.mp4}"
OUT="${OUT:-demo.mp4}"
FONT="${FONT:-/System/Library/Fonts/Supplemental/Arial.ttf}"
MONO="${MONO:-/System/Library/Fonts/Menlo.ttc}"

[ -f "$RAW" ] || { echo "no $RAW, run scripts/record-demo.sh first" >&2; exit 1; }

read -r W H FPS < <(ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate -of csv=p=0 "$RAW" \
  | awk -F, '{split($3,a,"/"); printf "%d %d %d\n", $1, $2, (a[2]?a[1]/a[2]:a[1])}')
echo "==> raw is ${W}x${H} @ ${FPS}fps"

card() { # text-file duration output
  local txt="$1" dur="$2" out="$3"
  ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i "color=c=0x1e1e2e:s=${W}x${H}:d=${dur}:r=${FPS}" \
    -vf "drawtext=fontfile=${FONT}:textfile=${txt}:fontcolor=0xcdd6f4:fontsize=$((H/22)):line_spacing=$((H/60)):x=(w-text_w)/2:y=(h-text_h)/2" \
    -c:v libx264 -pix_fmt yuv420p "$out"
}

cat > /tmp/vulcan-title.txt <<'EOF'
Vulcan

A codebase agent that never calls a third-party LLM API.
Generation runs on an AMD Radeon GPU via vLLM on ROCm.

AMD AI DevMaster Hackathon, Track 2, Private AI Agents
EOF

cat > /tmp/vulcan-end.txt <<'EOF'
Measured, not claimed

Concurrency 1 to 8 on the Radeon:  23.5 to 179.2 chunks/s
7.63x throughput at 95% scaling efficiency, TTFT 1.49s to 0.41s

Same harness on an M4 Air with Ollama:
one extra request LOWERS throughput, TTFT 4.36s to 42.21s

github.com/himanshu748/vulcan
EOF

echo "==> rendering cards"
card /tmp/vulcan-title.txt 7 /tmp/vulcan-title.mp4
card /tmp/vulcan-end.txt 9 /tmp/vulcan-end.mp4

echo "==> normalising the recording"
ffmpeg -hide_banner -loglevel error -y -i "$RAW" \
  -c:v libx264 -pix_fmt yuv420p -r "$FPS" -an /tmp/vulcan-body.mp4

echo "==> concatenating"
printf "file '%s'\n" /tmp/vulcan-title.mp4 /tmp/vulcan-body.mp4 /tmp/vulcan-end.mp4 > /tmp/vulcan-concat.txt
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i /tmp/vulcan-concat.txt -c copy "$OUT"

dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" | cut -d. -f1)
echo
echo "==> $OUT  ($((dur/60))m $((dur%60))s)"
ls -lh "$OUT"
[ "$dur" -ge 180 ] && [ "$dur" -le 300 ] \
  && echo "    inside the 3 to 5 minute window" \
  || echo "    WARNING: outside the 3 to 5 minute window, adjust sleeps in scripts/demo.tape"
