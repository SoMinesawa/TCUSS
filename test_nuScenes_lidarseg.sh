#!/bin/bash
# TCUSS: nuScenes LiDARSeg test submission script
#
# Usage:
#   ./test_nuScenes_lidarseg.sh <config_yaml>
#
# Example:
#   ./test_nuScenes_lidarseg.sh config/nuscenes_lidarseg_test.yaml

set -e

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <config_yaml>"
  exit 1
fi

CONFIG_YAML="$1"

python test_nuScenes_lidarseg.py --config "$CONFIG_YAML"



