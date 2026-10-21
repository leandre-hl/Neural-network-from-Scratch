#ifndef _H_ACTIVATIONS_
#define _H_ACTIVATIONS_


#include "../MatrixLib/MatrixLib.h"

double sigmoid(double input);
Matrix* sigmoidPrime(Matrix* m);
double ReLU(double x);
double derivative_ReLU(double x);
Matrix* softmax(Matrix* m);

#endif // _H_ACTIVATIONS_