#!/usr/bin/env bash

if [ "$#" -ne 3 ]; then
    echo "Usage: move-robot <x> <y> <a>"
fi

x_pos=$1
y_pos=$2
a_rot=$3

q_w=$(python -c "import math; print(math.cos($3/2))")
q_z=$(python -c "import math; print(math.sin($3/2))")

rostopic pub -1 /gazebo/set_model_state gazebo_msgs/ModelState "model_name: 'turtlebot$'
pose:
  position:
    x: $x_pos
    y: $y_pos
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: $q_z
    w: $q_w
"

