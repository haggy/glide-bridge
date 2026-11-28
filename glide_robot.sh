#!/bin/bash
# Laptop-side helper for the Glide robot (Glidance alpha 2) over SSH.
#
#   ./glide_robot.sh deploy           copy the relay to /data/glide_link on the robot
#   ./glide_robot.sh start | stop | status | log
#                                     run the relay in the background (port 8770)
#   ./glide_robot.sh info             active config, service states, disk space
#   ./glide_robot.sh uploader stop|start|status
#                                     Glidance's cloud bag uploader
#   ./glide_robot.sh bag start|stop   on-robot bag recording (data-collection mode only)
#   ./glide_robot.sh nav config | start | stop | status | log
#                                     data-collection mode + Glidance's local costmap:
#                                     config builds /data/glide_link/glide_alpha2_dc_nav.yaml
#                                     on the robot (robot/make_nav_config.py); start stops
#                                     glide_launch.service and launches that config instead;
#                                     stop ends it and restarts the service. Nav2 gets no
#                                     goals (goals_orchestrator off), so only the costmap runs.
#   ./glide_robot.sh bags             list bags in /data
#   ./glide_robot.sh pull-bags [N] [DEST]
#                                     copy the N newest finished bags (default 1) to DEST
#                                     (default ./data/bags)
#   ./glide_robot.sh ssh              interactive shell on the robot
#
# Run deploy again after changing anything in robot/.
#
# Robot address: GLIDE_HOST (default 10.42.0.45, room 209 hotspot; room 506 is
# 192.168.0.184). SSH connections are shared for 10 minutes, so a password is
# asked for at most once. Use ssh-copy-id root@$GLIDE_HOST to skip it entirely.
set -euo pipefail

HOST=${GLIDE_HOST:-10.42.0.45}
DEST_DIR=/data/glide_link
NAV_CFG=$DEST_DIR/glide_alpha2_dc_nav.yaml
HERE=$(cd "$(dirname "$0")" && pwd)
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/glide-%r@%h:%p" -o ControlPersist=10m
          -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new)
R() { ssh "${SSH_OPTS[@]}" "root@$HOST" "$@"; }
ROSENV='set +u; source /usr/bin/ros_setup.bash; cd /shared; source /shared/setup.bash;
export ROS_DOMAIN_ID=33 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///etc/cyclonedds/localhost_config.xml;'

cmd=${1:-help}; shift || true
case "$cmd" in
  deploy)
    R "mkdir -p $DEST_DIR"
    scp "${SSH_OPTS[@]}" -q "$HERE/robot/glide_link_relay.py" "$HERE/robot/run_relay.sh" \
        "$HERE/robot/make_nav_config.py" "root@$HOST:$DEST_DIR/"
    R "chmod +x $DEST_DIR/*.sh $DEST_DIR/*.py && ls -l $DEST_DIR"
    ;;
  start)
    R "pgrep -f glide_link_relay.py >/dev/null && { echo 'relay already running'; exit 0; }
       cd $DEST_DIR && nohup ./run_relay.sh $* > $DEST_DIR/relay.log 2>&1 < /dev/null &
       sleep 4; tail -n 5 $DEST_DIR/relay.log"
    echo "Relay: $HOST:8770"
    ;;
  stop)
    R "pkill -INT -f glide_link_relay.py && echo stopped || echo 'not running'"
    ;;
  status)
    R "pgrep -af glide_link_relay.py || echo 'relay not running'; tail -n 3 $DEST_DIR/relay.log 2>/dev/null || true"
    ;;
  log)
    R "tail -n ${1:-50} -f $DEST_DIR/relay.log"
    ;;
  info)
    R "echo -n 'boot config: '; grep -o 'config:=[^ ]*' /home/root/bringup.sh || true
       for s in glide_launch glidance-rosbag-uploader; do echo -n \"\$s: \"; systemctl is-active \$s.service || true; done
       df -h / /data | sed 1d
       echo -n 'relay: '; pgrep -f glide_link_relay.py >/dev/null && echo running || echo stopped
       echo -n 'dc+costmap launch: '; pgrep -f 'config:=$NAV_CFG' >/dev/null && echo running || echo stopped"
    ;;
  uploader)
    case "${1:-status}" in
      stop)   R "systemctl stop glidance-rosbag-uploader.service; systemctl is-active glidance-rosbag-uploader.service || true" ;;
      start)  R "systemctl start glidance-rosbag-uploader.service; systemctl is-active glidance-rosbag-uploader.service" ;;
      status) R "systemctl is-active glidance-rosbag-uploader.service || true" ;;
      *) echo "uploader stop|start|status"; exit 2 ;;
    esac
    ;;
  bag)
    case "${1:-}" in
      start) v=true ;; stop) v=false ;; *) echo "bag start|stop"; exit 2 ;;
    esac
    R "$ROSENV ros2 service call /bag_record std_srvs/srv/SetBool '{data: $v}'"
    ;;
  nav)
    case "${1:-status}" in
      config)
        R "python3 $DEST_DIR/make_nav_config.py --out $NAV_CFG"
        ;;
      start)
        R "set -e
           [ -f $NAV_CFG ] || python3 $DEST_DIR/make_nav_config.py --out $NAV_CFG
           pgrep -f 'config:=$NAV_CFG' >/dev/null && { echo 'dc+costmap launch already running'; exit 0; }
           systemctl stop glide_launch.service
           cd $DEST_DIR && nohup bash -c '$ROSENV exec ros2 launch glide_launchers glide.launch.py rviz:=false config:=$NAV_CFG release:=true' > $DEST_DIR/nav.log 2>&1 < /dev/null &
           sleep 5; tail -n 5 $DEST_DIR/nav.log"
        echo "Launched. Nav2 starts after its launch delay; check with: $0 nav status"
        ;;
      stop)
        R "if pkill -INT -f 'config:=$NAV_CFG'; then
             for i in \$(seq 20); do pgrep -f 'config:=$NAV_CFG' >/dev/null || break; sleep 1; done
             pgrep -f 'config:=$NAV_CFG' >/dev/null && { echo 'still running after 20 s; sending TERM'; pkill -TERM -f 'config:=$NAV_CFG'; sleep 3; }
             echo 'dc+costmap launch stopped'
           else echo 'dc+costmap launch not running'; fi
           systemctl start glide_launch.service || true
           echo -n 'glide_launch.service: '; systemctl is-active glide_launch.service || echo '(inactive: see the ALLOW_NAVIGATION flag in the troubleshooting log)'"
        ;;
      status)
        R "$ROSENV
           echo -n 'launch: '; pgrep -f 'config:=$NAV_CFG' >/dev/null && echo running || echo stopped
           echo -n 'glide_launch.service: '; systemctl is-active glide_launch.service || true
           echo '-- nodes:'; timeout 10 ros2 node list 2>/dev/null | grep -E 'controller_server|traversability|overhead_point|ir_filter|lifecycle' || echo '  (none of the costmap nodes)'
           echo '-- /local_costmap/costmap rate:'; timeout 8 ros2 topic hz /local_costmap/costmap 2>&1 | grep -m2 -E 'average|WARNING' || echo '  no messages in 8 s'
           echo '-- TF map -> base_footprint (costmap global_frame is map):'; timeout 5 ros2 run tf2_ros tf2_echo map base_footprint 2>&1 | grep -m3 -E 'Translation|Rotation|exist|Invalid' || echo '  no transform in 5 s'"
        ;;
      log)
        R "tail -n ${2:-50} -f $DEST_DIR/nav.log"
        ;;
      *) echo "nav config|start|stop|status|log"; exit 2 ;;
    esac
    ;;
  bags)
    R "cd /data && for d in \$(ls -dt rosbag2_* 2>/dev/null); do echo \"\$(du -sh \$d | cut -f1)  \$d\"; done"
    ;;
  pull-bags)
    n=${1:-1}; dest=${2:-$HERE/data/bags}; mkdir -p "$dest"
    # newest N directories that have metadata.yaml (written when a bag is closed)
    bags=$(R "cd /data && for d in \$(ls -dt rosbag2_* 2>/dev/null); do [ -f \$d/metadata.yaml ] && echo \$d; done | head -n $n")
    [ -z "$bags" ] && { echo "no finished bags in /data"; exit 1; }
    for b in $bags; do
      echo "pulling $b -> $dest/$b"
      R "tar -C /data -cf - $b" | tar -C "$dest" -xf -
      du -sh "$dest/$b"
    done
    ;;
  ssh)
    ssh "${SSH_OPTS[@]}" "root@$HOST"
    ;;
  *)
    sed -n '2,28p' "$0"; exit 2 ;;
esac
