NV_GPU=6  \
nvidia-docker run --rm -t -i \
-v /mnt/data:/mnt/data \
-v /mnt/iscsi:/mnt/fast \
-v /home/hhofmann/code:/workspace/code \
-u $(id -u):$(id -g) \
nvcr.io/nvidia/tensorflow:17.09 \
python code/pretrainingYolo/pretraining.py  --name $1 \
                                            --learningrate 0.001 \
                                            --batchSize 128 \
                                            --numThreads 16 \
                                            --bufferSize 1300000 \
                                            --originPath "/mnt/data/ilsvrc2012/LabelList_Heinz/" \
                                            --nrOfEpochs 10000000 \
                                            --nrOfEpochsUntilSaveModel 1000
#for make directly something inside the docker-container, make a backslash after 17.09 and type the command you want to have processed on a new line. 