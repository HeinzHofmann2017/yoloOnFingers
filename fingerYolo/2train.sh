NV_GPU=2  \
nvidia-docker run --rm -t -i \
-v /mnt/data:/mnt/data \
-v /mnt/iscsi:/mnt/fast \
-v /home/hhofmann/code:/workspace/code \
-u $(id -u):$(id -g) \
nvcr.io/nvidia/tensorflow:17.09 \
python code/fingerYolo/training.py          --name $1 \
                                            --learningrate 0.00001 \
                                            --batchSize 7 \
                                            --numThreads 16 \
                                            --bufferSize 1000 \
                                            --originPath "/mnt/data/data_hhofmann/Data/indexfinger_right/3000_readyTOlearn/trainData/" \
                                            --nrOfEpochs 10000000 \
                                            --nrOfEpochsUntilSaveModel 1
#for make directly something inside the docker-container, make a backslash after 17.09 and type the command you want to have processed on a new line. 
