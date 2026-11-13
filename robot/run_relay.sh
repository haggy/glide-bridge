#!/bin/bash
# Start the glide_link relay on the Glide robot with the same ROS environment
# as glide_launch.service (domain 33, Cyclone DDS bound to localhost).
# For the CARLA overlays, which use Fast DDS, run with:
#   RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
#   FASTRTPS_DEFAULT_PROFILES_FILE=/shared/glide_carla/fastdds_robot_wifi_profile.xml ./run_relay.sh
set +u
source /usr/bin/ros_setup.bash
cd /shared && source /shared/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-33}
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}
if [ "$RMW_IMPLEMENTATION" = rmw_cyclonedds_cpp ]; then
  export CYCLONEDDS_URI=${CYCLONEDDS_URI:-file:///etc/cyclonedds/localhost_config.xml}
fi
exec /usr/bin/python3 "$(dirname "$(readlink -f "$0")")/glide_link_relay.py" "$@"
