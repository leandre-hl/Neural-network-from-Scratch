#include "./neural_network.h"
#include "../parameters.h"
#include "./activations.c"
#include "../util/img.c"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static char* network_file_path(const char* dir_path, const char* file_name) {
    size_t path_size = strlen(dir_path) + strlen(file_name) + 2;
    char* path = malloc(path_size);
    if (path == NULL) {
        fprintf(stderr, "Neural network error: out of memory\n");
        exit(EXIT_FAILURE);
    }
    snprintf(path, path_size, "%s/%s", dir_path, file_name);
    return path;
}

NeuralNetwork* network_create(int input, int hidden, int output, double lr) {
    NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
    net->input = input;
    net->hidden = hidden;
    net->output = output;
    net->learning_rate = lr;
    net->hidden_weights = matrix_create(hidden, input);
    net->output_weights = matrix_create(output, hidden);
    net->hidden_biases = matrix_create(hidden, 1);
    net->output_biases = matrix_create(output, 1);
    matrix_randomize(net->hidden_weights, hidden);
    matrix_randomize(net->output_weights, output);
    matrix_randomize(net->hidden_biases, hidden);
    matrix_randomize(net->output_biases, output);
    
    return net;
}

void network_train(NeuralNetwork* net, Matrix* input_data, Matrix* output_data) {
    // Forward propagation
    Matrix* W1_dot_A0 = dot(net->hidden_weights, input_data);
    Matrix* Z1 = add(W1_dot_A0, net->hidden_biases);
    Matrix* A1 = apply(ReLU, Z1);
    Matrix* W2_dot_A1 = dot(net->output_weights, A1);
    Matrix* Z2 = add(W2_dot_A1, net->output_biases);
    Matrix* A2 = softmax(Z2);

    // Back propagation
    double m = input_data->cols;
    // dZ2 = A2 - Y
    Matrix* dZ2 = subtract(A2, output_data);
    // dW2 = alpha/m * dZ2.A1T 
    Matrix* A1T = transpose(A1);
    Matrix* dZ2_dot_A1T = dot(dZ2, A1T);
    Matrix* alpha_dW2 = scale(net->learning_rate/m, dZ2_dot_A1T);
    // db2 = alpha/m * sum(dZ2)
    Matrix* sum_dZ2 = sum(dZ2, 0);
    Matrix* alpha_db2 = scale(net->learning_rate/m, sum_dZ2);
    // dZ1 = W2T.dZ2 * g'(Z1)
    Matrix* W2T = transpose(net->output_weights);
    Matrix* W2T_dot_dZ2 = dot(W2T, dZ2);

    Matrix* deriv_ReLU_Z1 = apply(derivative_ReLU, Z1);
    Matrix* dZ1 = multiply(W2T_dot_dZ2, deriv_ReLU_Z1);
    // dW1 = alpha/m * dZ1.A0T
    Matrix* A0T = transpose(input_data);
    Matrix* dZ1_dot_A0T = dot(dZ1, A0T);
    Matrix* alpha_dW1 = scale(net->learning_rate/m, dZ1_dot_A0T);
    // db1 = 1/m * sum(dZ1)
    Matrix* sum_dZ1 = sum(dZ1, 0);
    Matrix* alpha_db1 = scale(net->learning_rate/m, sum_dZ1);
    
    // Adjust weights and biases
    Matrix* new_W1 = subtract(net->hidden_weights, alpha_dW1);
    Matrix* new_b1 = subtract(net->hidden_biases, alpha_db1);
    Matrix* new_W2 = subtract(net->output_weights, alpha_dW2);
    Matrix* new_b2 = subtract(net->output_biases, alpha_db2);
    matrix_free(net->hidden_weights); // Free the old weights and biases before replacement
    matrix_free(net->hidden_biases);
    matrix_free(net->output_weights);
    matrix_free(net->output_biases);
    net->hidden_weights = new_W1;
    net->hidden_biases = new_b1;
    net->output_weights = new_W2;
    net->output_biases = new_b2;

    // Free all matrices
    matrix_free(W1_dot_A0);
    matrix_free(Z1);
    matrix_free(A1);
    matrix_free(W2_dot_A1);
    matrix_free(Z2);
    matrix_free(A2);
    matrix_free(dZ2);
    matrix_free(A1T);
    matrix_free(dZ2_dot_A1T);
    matrix_free(alpha_dW2);
    matrix_free(sum_dZ2);
    matrix_free(alpha_db2);
    matrix_free(W2T);
    matrix_free(W2T_dot_dZ2);
    matrix_free(deriv_ReLU_Z1);
    matrix_free(dZ1);
    matrix_free(A0T);
    matrix_free(dZ1_dot_A0T);
    matrix_free(alpha_dW1);
    matrix_free(sum_dZ1);
    matrix_free(alpha_db1);
}

void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size) {
    Matrix* input_data = NULL; // Mx1 matrix
    Matrix* output_data = matrix_create(net->output, 1); // Nx1 matrix
    for (int i = 0; i < batch_size; i++) {
        printf("Image No. %d | Current progress : %1.1f %c\n", i, 100.0 * i/batch_size, '%');
        input_data = imgs[i]->img_data;
        matrix_fill(output_data, 0);
        output_data->entries[imgs[i]->label][0] = 1; // Setting the result, ex : [0, 0, 1, 0, ..., 0]
        network_train(net, input_data, output_data);
    }
    printf("Image No. %d | Current progress : 100.0 %c\n", batch_size, '%');
    matrix_free(output_data);
}

Matrix* network_predict(NeuralNetwork* net, Matrix* input_data) {
    Matrix* W1_dot_A0 = dot(net->hidden_weights, input_data);
    Matrix* Z1 = add(W1_dot_A0, net->hidden_biases);
    Matrix* A1 = apply(ReLU, Z1);
    Matrix* W2_dot_A1 = dot(net->output_weights, A1);
    Matrix* Z2 = add(W2_dot_A1, net->output_biases);
    Matrix* A2 = softmax(Z2);
    
    matrix_free(W1_dot_A0);
    matrix_free(Z1);
    matrix_free(A1);
    matrix_free(W2_dot_A1);
    matrix_free(Z2);
    
    return A2;
}

Matrix* network_predict_img(NeuralNetwork* net, Img* img) {
	return network_predict(net, img->img_data); 
}

double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n) {
	int n_correct = 0;
	for (int i = 0; i < n; i++) {
		Matrix* prediction = network_predict_img(net, imgs[i]);
		if (matrix_argmax(prediction) == imgs[i]->label) {
			n_correct++;
		}
		matrix_free(prediction);
	}
	return 1.0 * n_correct / n;
}

void network_save(NeuralNetwork* net, char* dir_path_str) {
    if (mkdir(dir_path_str, 0755) != 0 && errno != EEXIST) {
        perror("Unable to create neural network directory");
        exit(EXIT_FAILURE);
    }

    char* path = network_file_path(dir_path_str, "descriptor");
    FILE* file = fopen(path, "w");
    if (file == NULL) {
        perror("Unable to open neural network descriptor for writing");
        free(path);
        exit(EXIT_FAILURE);
    }
    fprintf(file, "%d\n", net->input);
    fprintf(file, "%d\n", net->hidden);
    fprintf(file, "%d\n", net->output);
    fprintf(file, "%.17g\n", net->learning_rate);
    fclose(file);
    free(path);

    path = network_file_path(dir_path_str, "hidden_weights");
    matrix_save(net->hidden_weights, path);
    free(path);
    path = network_file_path(dir_path_str, "hidden_biases");
    matrix_save(net->hidden_biases, path);
    free(path);
    path = network_file_path(dir_path_str, "output_weights");
    matrix_save(net->output_weights, path);
    free(path);
    path = network_file_path(dir_path_str, "output_biases");
    matrix_save(net->output_biases, path);
    free(path);

    printf("Successfully saved neural network to %s\n", dir_path_str);
}

NeuralNetwork* network_load(char* dir_path_str) {
    char* path = network_file_path(dir_path_str, "descriptor");
    FILE* file = fopen(path, "r");
    if (file == NULL) {
        perror("Unable to open neural network descriptor for reading");
        free(path);
        exit(EXIT_FAILURE);
    }
    NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
    if (net == NULL) {
        fprintf(stderr, "Neural network error: out of memory\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (fscanf(file, "%d%d%d%lf", &net->input, &net->hidden, &net->output,
               &net->learning_rate) != 4 ||
        net->input <= 0 || net->hidden <= 0 || net->output <= 0) {
        fprintf(stderr, "Neural network error: invalid descriptor in '%s'\n", path);
        fclose(file);
        free(path);
        free(net);
        exit(EXIT_FAILURE);
    }
    fclose(file);
    free(path);

    path = network_file_path(dir_path_str, "hidden_weights");
    net->hidden_weights = matrix_load(path);
    free(path);
    path = network_file_path(dir_path_str, "hidden_biases");
    net->hidden_biases = matrix_load(path);
    free(path);
    path = network_file_path(dir_path_str, "output_weights");
    net->output_weights = matrix_load(path);
    free(path);
    path = network_file_path(dir_path_str, "output_biases");
    net->output_biases = matrix_load(path);
    free(path);

    if (net->hidden_weights->rows != net->hidden || net->hidden_weights->cols != net->input ||
        net->hidden_biases->rows != net->hidden || net->hidden_biases->cols != 1 ||
        net->output_weights->rows != net->output || net->output_weights->cols != net->hidden ||
        net->output_biases->rows != net->output || net->output_biases->cols != 1) {
        fprintf(stderr, "Neural network error: saved matrix dimensions do not match descriptor\n");
        network_free(net);
        exit(EXIT_FAILURE);
    }
    printf("Successfully loaded neural_network from %s\n", dir_path_str);
    return net;
}

void network_print(NeuralNetwork* net) {
    printf("Input layer : %d\n", net->input);
    printf("Hidden layer : %d\n", net->hidden);
    printf("Output layer : %d\n", net->output);
    printf("Learning rate : %1.6f\n", net->learning_rate);
    printf("==============\n");
    printf("Hidden weights\n");
    printf("==============\n");
    matrix_print(net->hidden_weights);
    printf("==============\n");
    printf("Hidden biases\n");
    printf("==============\n");
    matrix_print(net->hidden_biases);
    printf("==============\n");
    printf("Output weights\n");
    printf("==============\n");
    matrix_print(net->output_weights);
    printf("==============\n");
    printf("Output biases\n");
    printf("==============\n");
    matrix_print(net->output_biases);
}

void network_free(NeuralNetwork *net) {
    if (net == NULL) return;
	matrix_free(net->hidden_weights);
    matrix_free(net->hidden_biases);
	matrix_free(net->output_weights);
	matrix_free(net->output_biases);
	free(net);
	net = NULL;
}