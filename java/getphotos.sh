
rm images/train/empty/*
rm images/train/intruder/*
rm images/test/empty/*
rm images/test/intruder/*
rm images/valid/empty/*
rm images/valid/intruder/*

rm images/infer/empty/images/*
rm images/infer/intruder/images/*
rm images/infer/intruder_traces/images/*


# TRAIN
java -jar copyRandFiles.jar COPY 834 ../images/empty/empty-0613 ./images/train/empty
java -jar copyRandFiles.jar COPY 834 ../images/empty/empty-0614 ./images/train/empty
java -jar copyRandFiles.jar COPY 834 ../images/empty/empty-0615 ./images/train/empty
java -jar copyRandFiles.jar COPY 834 ../images/empty/empty-0616 ./images/train/empty
java -jar copyRandFiles.jar COPY 834 ../images/empty/empty-0617 ./images/train/empty
java -jar copyRandFiles.jar COPY 834 ../images/empty/empty-0618 ./images/train/empty

java -jar copyRandFiles.jar COPY 4702 ../images/intruder/full_body ./images/train/intruder
java -jar copyRandFiles.jar COPY 100 ../images/intruder/door ./images/train/intruder
java -jar copyRandFiles.jar COPY 200 ../images/intruder/night ./images/train/intruder



# VALIDATE
java -jar copyRandFiles.jar MOVE 800 ./images/train/empty ./images/valid/empty

java -jar copyRandFiles.jar MOVE 800 ./images/train/intruder ./images/valid/intruder




# TEST
java -jar copyRandFiles.jar MOVE 800 ./images/train/empty ./images/test/empty

java -jar copyRandFiles.jar MOVE 800 ./images/train/intruder ./images/test/intruder



# INFER
java -jar copyRandFiles.jar MOVE 1000 ./images/train/empty ./images/infer/empty/images
java -jar copyRandFiles.jar MOVE 1000 ./images/train/intruder ./images/infer/intruder/images
java -jar copyRandFiles.jar COPY 1000 ../images/intruder/traces ./images/infer/intruder_traces/images

