#include "activations.h"

#include <math.h>
#include "../MatrixLib/MatrixLib.c"

double sigmoid(double input) {
	return 1.0 / (1 + exp(-1 * input));
}

Matrix* sigmoidPrime(Matrix* m) {
	Matrix* ones = matrix_create(m->rows, m->cols);
	matrix_fill(ones, 1);
	Matrix* subtracted = subtract(ones, m);
	Matrix* multiplied = multiply(m, subtracted);
	matrix_free(ones);
	matrix_free(subtracted);
	return multiplied;
}

double ReLU(double x) {
    return x > 0 ? x : 0;
}

double derivative_ReLU(double x) {
    return x > 0;
}

Matrix* softmax(Matrix* m) {
	Matrix* mat = matrix_create(m->rows, m->cols);
	for (int j = 0; j < m->cols; j++) {
		double max_value = m->entries[0][j];
		for (int i = 1; i < m->rows; i++) {
			if (m->entries[i][j] > max_value) {
				max_value = m->entries[i][j];
			}
		}

		double total = 0;
		for (int i = 0; i < m->rows; i++) {
			mat->entries[i][j] = exp(m->entries[i][j] - max_value);
			total += mat->entries[i][j];
		}
		for (int i = 0; i < m->rows; i++) {
			mat->entries[i][j] /= total;
		}
	}
	return mat;
}