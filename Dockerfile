version: '3.11'

services:
  neutron-reflectivity:
    build: .
    volumes:
      - .:/app
    environment:
      - DISPLAY=${DISPLAY}
      - QT_X11_NO_MITSHM=1
    devices:
      - /dev/snd:/dev/snd
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix
    command: python main.py