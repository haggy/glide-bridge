# Source (don't run) this to enter the ROS 2 Humble env for glide_bridge/ros/:
#   source env/activate_ros.sh
# First time: conda env create -f env/ros_humble.yml && source env/activate_ros.sh && pip install -e . --no-deps
_conda_base="$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")"
# conda's and RoboStack's activation hooks read unset variables (e.g. $CONDA_BUILD),
# which kills callers running under `set -u`. Relax nounset while activating, then restore.
case $- in *u*) _had_nounset=1; set +u ;; *) _had_nounset=0 ;; esac
# Leave the uv venv first: if .venv/bin stays ahead on PATH, `python` is the venv's 3.12
# while ROS points PYTHONPATH at the env's 3.11 packages, and numpy's C extension fails
# to import. Child shells (e.g. stream_demo.sh) inherit PATH but not the `deactivate`
# function, so strip the venv from PATH by hand as well.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    [ "$(type -t deactivate)" = function ] && deactivate
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        PATH=":$PATH:"; PATH="${PATH//:$VIRTUAL_ENV\/bin:/:}"; PATH="${PATH#:}"; PATH="${PATH%:}"
        export PATH
        unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
    fi
fi
source "$_conda_base/etc/profile.d/conda.sh"
conda activate glide_ros
[ "$_had_nounset" = 1 ] && set -u
unset _had_nounset
# The robot's stack uses Cyclone DDS; match it so the later switch to live data is just a topic change.
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# Keep replayed camera traffic off the Wi-Fi. After sourcing, set this to 0 (and set ROS_DOMAIN_ID=33,
# the robot's domain) when talking to a ROS graph on another machine.
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
unset _conda_base
echo "glide_ros: ROS_DISTRO=$ROS_DISTRO RMW=$RMW_IMPLEMENTATION DOMAIN=$ROS_DOMAIN_ID LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY"
