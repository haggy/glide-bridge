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
#   ./glide_robot.sh bags             list bags in /data
#   ./glide_robot.sh pull-bags [N] [DEST]
#                                     copy the N newest finished bags (default 1) to DEST
#                                     (default ./data/bags)
#   ./glide_robot.sh ssh              interactive shell on the robot
#
# Robot address: GLIDE_HOST (default 10.42.0.45, room 209 hotspot; room 506 is
# 192.168.0.184). SSH connections are shared for 10 minutes, so a password is
# asked for at most once. Use ssh-copy-id root@$GLIDE_HOST to skip it entirely.
set -euo pipefail

HOST=${GLIDE_HOST:-10.42.0.45}
DEST_DIR=/data/glide_link
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
    scp "${SSH_OPTS[@]}" -q "$HERE/robot/glide_link_relay.py" "$HERE/robot/run_relay.sh" "root@$HOST:$DEST_DIR/"
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
       echo -n 'relay: '; pgrep -f glide_link_relay.py >/dev/null && echo running || echo stopped"
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
    sed -n '2,20p' "$0"; exit 2 ;;
esac
