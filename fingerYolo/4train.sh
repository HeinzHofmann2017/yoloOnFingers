NV_GPU=4  \
nvidia-docker run --rm -t -i \
-v /mnt/data:/mnt/data \
-v /mnt/iscsi:/mnt/fast \
-v /home/hhofmann/code:/workspace/code \
-u $(id -u):$(id -g) \
nvcr.io/nvidia/tensorflow:17.09 \
python code/fingerYolo/training.py          --name $1 \
                                            --learningrate 0.001 \
                                            --batchSize 4 \
                                            --numThreads 16 \
                                            --bufferSize 1000 \
                                            --originPath "/mnt/data/data_hhofmann/Data/indexfinger_right/3000_readyTOlearn/trainData/" \
                                            --nrOfEpochs 10000000 \
                                            --nrOfEpochsUntilSaveModel 10 \
                                            --randSeed $2
#for make directly something inside the docker-container, make a backslash after 17.09 and type the command you want to have processed on a new line. 
