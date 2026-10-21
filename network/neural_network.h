#ifndef _H_NEURAL_NETWORK_
#define _H_NEURAL_NETWORK_

#include "../MatrixLib/MatrixLib.h"
#include "../util/img.h"

typedef struct {
    int input;
    int hidden;
    int output;
    double learning_rate;
    Matrix* hidden_weights;
    Matrix* hidden_biases;
    Matrix* output_weights;
    Matrix* output_biases;
} NeuralNetwork;

NeuralNetwork* network_create(int input, int hidden, int output, double lr);
void network_train(NeuralNetwork* net, Matrix* input_data, Matrix* output_data);
void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size);
Matrix* network_predict(NeuralNetwork* net, Matrix* input_data);
Matrix* network_predict_img(NeuralNetwork* net, Img* img);
double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n);
void network_save(NeuralNetwork* net, char* dir_path_str);
NeuralNetwork* network_load(char* dir_path_str);
void network_print(NeuralNetwork* net);
void network_free(NeuralNetwork* net);

#endif // _H_NEURAL_NETWORK_