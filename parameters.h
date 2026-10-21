#ifndef _H_PARAMETERS_
#define _H_PARAMETERS_

#define IMG_ROWS 28
#define IMG_COLS 28
#define IMG_PIXELS 784

#define MAX_MNIST_CHAR 10000

#define TRAIN_DATASET_PATH "./data/mnist_train.csv"
#define TEST_DATASET_PATH  "./data/mnist_test.csv"
#define USED_DATASET_PATH    TRAIN_DATASET_PATH


#define LOAD_DIR_PATH "network_save"
#define SAVE_DIR_PATH "network_save2"

#define INPUT 784
#define HIDDEN 500
#define OUTPUT 10
#define LEARNING_RATE 0.02

#define N_TRAIN_IMGS 2000
#define N_TEST_IMGS 1000
//#define MAX_LOAD 100

#endif // _H_PARAMETERS_