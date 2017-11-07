NV_GPU=2  \
nvidia-docker run --rm -t -i \
-v /home/hhofmann/data:/mnt/data \
-v /mnt/iscsi:/mnt/fast \
-v /home/hhofmann/code:/workspace/code \
-u $(id -u):$(id -g) \
nvcr.io/nvidia/tensorflow:17.09 \
python code/pretrainingYolo/pretraining.py
#for make directly something inside the docker-container, make a backslash after 17.09 and type the command you want to have processed on a new line. 
