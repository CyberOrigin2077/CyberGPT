sudo docker run -it --name automerge --gpus=all -p 5900:5900 -v /home/$USER/codespace/AutoMerge:/root/work/AutoMerge -v $PITT_DATA:/root/data metalsam/automerge:latest
