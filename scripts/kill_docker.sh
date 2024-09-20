#!/usr/bin/env bash

docker stop $(docker ps -aq)

# docker container prune   # Remove all stopped containers
# docker volume prune      # Remove all unused volumes
# docker image prune       # Remove unused images
# docker system prune      # All of the above, in this order: containers, volumes, images
echo y | docker system prune