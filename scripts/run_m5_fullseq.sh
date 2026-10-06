#!/bin/bash
set -e
cd /d/yizhen/smart_kitchen
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python.exe
ROOT=D:/yizhen/CHIRLA/CHIRLA_data/CHIRLA
SEQS="seq_004 seq_006 seq_007 seq_020 seq_024 seq_025 seq_026"
NAME=$1; TOPO=$2; EMB=$3
for seq in $SEQS; do
  VIDS=""
  CAMS=""
  for c in camera_1 camera_2 camera_3 camera_4 camera_5 camera_6 camera_7; do
    v=$(ls $ROOT/clips_singal_person_result/$seq/${c}_*.avi 2>/dev/null | head -1)
    [ -n "$v" ] && VIDS="$VIDS $v" && CAMS="$CAMS $c"
  done
  $PY scripts/m5_track_video.py --videos $VIDS --cameras $CAMS \
    --topology "$TOPO" --tracker configs/tracker_rf_cbiou.yaml \
    --variant nano --thr 0.10 --person-cls 1 --embedder "$EMB" --fps 30.0 \
    --max-frames -1 --ttl 600 --stride 5 \
    --det-cache results/det_cache/coco_nano/$seq \
    --out results/m5_topofix/$NAME/$seq/chef_events.jsonl > /dev/null 2>&1
  echo "  $NAME $seq 完成"
done
$PY scripts/eval_m4m5_chirla.py --root $ROOT \
  --run-dir $(for s in $SEQS; do echo -n "results/m5_topofix/$NAME/$s "; done) \
  --seq $SEQS --topology "$TOPO" --out results/m5_topofix/$NAME 2>&1 | tail -3
