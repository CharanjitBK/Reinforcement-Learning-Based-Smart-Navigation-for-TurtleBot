xhost +local:docker

docker run -it \
  --gpus all \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --env="XAUTHORITY=/root/.Xauthority" \
  --env="LIBGL_ALWAYS_SOFTWARE=0" \
  --env="NO_AT_BRIDGE=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
  --device=/dev/dri:/dev/dri \
  --net=host \
  --name ros-foxy \
  --rm \
  ubuntu-20 \
  bash
