#!/usr/bin/env bash
set -euo pipefail
R=/storage_fast/ttu/GenPlaylist
E=$R/experiments/tiger-joint-20261108
DS=${1:?}; START=${2:?}; COUNT=${3:?}
bash "$E/audio_joint_v2.sh" "$DS" full "$START" "$COUNT"
"$R/envs/ace-step/bin/python" "$E/validate_joint_v2.py" --dataset "$DS" --start "$START" --count "$COUNT"
