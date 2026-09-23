#!/usr/bin/env bash
# Builds maze_vla_ros with the venv active so console-script shebangs point at
# venv/bin/python3 (which has torch/clip/gymnasium) instead of /usr/bin/python3.
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

source venv/bin/activate
source /opt/ros/jazzy/setup.bash

colcon build --packages-select maze_vla_ros "$@"
