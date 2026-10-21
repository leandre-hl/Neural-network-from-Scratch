#include <stdio.h>
#include <stdlib.h>

#include "./parameters.h"
#include "./network/neural_network.c"

// TODO : Check if each library is only downloaded once in the code
// TODO : Check the spelling in all the files
// TODO : Check if each function is defined in header files
// TODO : Fix the issue with saving learning rate
// TODO : Fix the issue with the saving directory included in the other

int main(int argc, char *argv[])
{
    // Create a new network and test its performances
    NeuralNetwork* net = network_create(INPUT, HIDDEN, OUTPUT, LEARNING_RATE);
    Img** imgs_test = csv_to_imgs(TEST_DATASET_PATH, 0, N_TEST_IMGS);
    printf("Accuracy : %1.1f %c\n", 100*network_predict_imgs(net, imgs_test, N_TEST_IMGS), '%');
    network_save(net, LOAD_DIR_PATH);
    imgs_free(imgs_test, N_TEST_IMGS);
    network_free(net);

    // Train the network
    /*
    Img** imgs_train = csv_to_imgs(TRAIN_DATASET_PATH, 0, N_TRAIN_IMGS);
    Img** imgs_test = csv_to_imgs(TEST_DATASET_PATH, 0, N_TEST_IMGS);
    NeuralNetwork* net = network_load(LOAD_DIR_PATH);
    network_train_batch_imgs(net, imgs_train, N_TRAIN_IMGS);
    printf("Accuracy : %1.1f %c\n", 100*network_predict_imgs(net, imgs_test, N_TEST_IMGS), '%');
    network_save(net, SAVE_DIR_PATH);
    imgs_free(imgs_train, N_TRAIN_IMGS);
    imgs_free(imgs_test, N_TEST_IMGS);
    network_free(net);
    */
    
    return 0;
}