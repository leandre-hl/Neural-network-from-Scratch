#ifndef _H_MATRIXLIB_
#define _H_MATRIXLIB_

// Maximum size of a line read from a descriptor file
#define MAX_CHAR 100

typedef struct {
    double** entries;   // entries[i][j] : row i, column j
    int rows;
    int cols;
} Matrix;

// ---------- Creation / destruction ----------
Matrix* matrix_create(int rows, int cols);   // zero-initialised
void    matrix_free(Matrix* m);
void    matrix_fill(Matrix* m, double value);
// Fills m with uniform random values in [-1/sqrt(n), 1/sqrt(n)]
void    matrix_randomize(Matrix* m, int n);

// ---------- Operations (all return a newly allocated matrix) ----------
Matrix* add(Matrix* a, Matrix* b);           // a + b (same dimensions)
Matrix* subtract(Matrix* a, Matrix* b);      // a - b (same dimensions)
Matrix* multiply(Matrix* a, Matrix* b);      // element-wise product (same dimensions)
Matrix* dot(Matrix* a, Matrix* b);           // matrix product (a->cols == b->rows)
Matrix* scale(double n, Matrix* m);          // n * m
Matrix* transpose(Matrix* m);
Matrix* apply(double (*func)(double), Matrix* m);  // func applied to every entry
// axis 0 : sums each row  -> matrix of size rows x 1
// axis 1 : sums each column -> matrix of size 1 x cols
Matrix* sum(Matrix* m, int axis);

// Index (row-major) of the largest entry; for a column vector, this is the row index
int     matrix_argmax(Matrix* m);

// ---------- Input / output ----------
void    matrix_print(Matrix* m);
void    matrix_save(Matrix* m, char* file_path);
Matrix* matrix_load(char* file_path);

#endif // _H_MATRIXLIB_