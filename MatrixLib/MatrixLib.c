#ifndef _C_MATRIXLIB_
#define _C_MATRIXLIB_

#include "MatrixLib.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ---------- Internal helpers ----------

static void matrix_error(const char* message) {
    fprintf(stderr, "MatrixLib error : %s\n", message);
    exit(EXIT_FAILURE);
}

static void check_same_dimensions(Matrix* a, Matrix* b, const char* op) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "MatrixLib error : %s expects equal dimensions, got %dx%d and %dx%d\n",
                op, a->rows, a->cols, b->rows, b->cols);
        exit(EXIT_FAILURE);
    }
}

// ---------- Creation / destruction ----------

Matrix* matrix_create(int rows, int cols) {
    if (rows <= 0 || cols <= 0) matrix_error("matrix_create expects positive dimensions");
    Matrix* m = malloc(sizeof(Matrix));
    if (m == NULL) matrix_error("out of memory");
    m->rows = rows;
    m->cols = cols;
    m->entries = malloc(rows * sizeof(double*));
    if (m->entries == NULL) matrix_error("out of memory");
    for (int i = 0; i < rows; i++) {
        m->entries[i] = calloc(cols, sizeof(double));
        if (m->entries[i] == NULL) matrix_error("out of memory");
    }
    return m;
}

void matrix_free(Matrix* m) {
    if (m == NULL) return;
    for (int i = 0; i < m->rows; i++) {
        free(m->entries[i]);
    }
    free(m->entries);
    free(m);
}

void matrix_fill(Matrix* m, double value) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->entries[i][j] = value;
        }
    }
}

void matrix_randomize(Matrix* m, int n) {
    double limit = 1.0 / sqrt((double) n);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            double r = (double) rand() / RAND_MAX;   // in [0, 1]
            m->entries[i][j] = -limit + 2.0 * limit * r;
        }
    }
}

// ---------- Operations ----------

Matrix* add(Matrix* a, Matrix* b) {
    check_same_dimensions(a, b, "add");
    Matrix* r = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            r->entries[i][j] = a->entries[i][j] + b->entries[i][j];
        }
    }
    return r;
}

Matrix* subtract(Matrix* a, Matrix* b) {
    check_same_dimensions(a, b, "subtract");
    Matrix* r = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            r->entries[i][j] = a->entries[i][j] - b->entries[i][j];
        }
    }
    return r;
}

Matrix* multiply(Matrix* a, Matrix* b) {
    check_same_dimensions(a, b, "multiply");
    Matrix* r = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            r->entries[i][j] = a->entries[i][j] * b->entries[i][j];
        }
    }
    return r;
}

Matrix* dot(Matrix* a, Matrix* b) {
    if (a->cols != b->rows) {
        fprintf(stderr, "MatrixLib error : dot expects a->cols == b->rows, got %dx%d and %dx%d\n",
                a->rows, a->cols, b->rows, b->cols);
        exit(EXIT_FAILURE);
    }
    Matrix* r = matrix_create(a->rows, b->cols);
    // i-k-j loop order : walks both b and r row by row (cache-friendly)
    for (int i = 0; i < a->rows; i++) {
        for (int k = 0; k < a->cols; k++) {
            double aik = a->entries[i][k];
            for (int j = 0; j < b->cols; j++) {
                r->entries[i][j] += aik * b->entries[k][j];
            }
        }
    }
    return r;
}

Matrix* scale(double n, Matrix* m) {
    Matrix* r = matrix_create(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            r->entries[i][j] = n * m->entries[i][j];
        }
    }
    return r;
}

Matrix* transpose(Matrix* m) {
    Matrix* r = matrix_create(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            r->entries[j][i] = m->entries[i][j];
        }
    }
    return r;
}

Matrix* apply(double (*func)(double), Matrix* m) {
    Matrix* r = matrix_create(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            r->entries[i][j] = func(m->entries[i][j]);
        }
    }
    return r;
}

Matrix* sum(Matrix* m, int axis) {
    Matrix* r;
    if (axis == 0) {
        r = matrix_create(m->rows, 1);
        for (int i = 0; i < m->rows; i++) {
            for (int j = 0; j < m->cols; j++) {
                r->entries[i][0] += m->entries[i][j];
            }
        }
    } else if (axis == 1) {
        r = matrix_create(1, m->cols);
        for (int i = 0; i < m->rows; i++) {
            for (int j = 0; j < m->cols; j++) {
                r->entries[0][j] += m->entries[i][j];
            }
        }
    } else {
        matrix_error("sum expects axis 0 or 1");
        return NULL;
    }
    return r;
}

int matrix_argmax(Matrix* m) {
    double best = m->entries[0][0];
    int best_index = 0;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (m->entries[i][j] > best) {
                best = m->entries[i][j];
                best_index = i * m->cols + j;
            }
        }
    }
    return best_index;
}

// ---------- Input / output ----------

void matrix_print(Matrix* m) {
    printf("Rows : %d Columns : %d\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%1.3f ", m->entries[i][j]);
        }
        printf("\n");
    }
}

// File format : "rows cols" on the first line, then one line per row
void matrix_save(Matrix* m, char* file_path) {
    FILE* file = fopen(file_path, "w");
    if (file == NULL) {
        fprintf(stderr, "MatrixLib error : cannot open '%s' for writing\n", file_path);
        exit(EXIT_FAILURE);
    }
    fprintf(file, "%d %d\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            fprintf(file, "%.17g%c", m->entries[i][j], j == m->cols - 1 ? '\n' : ' ');
        }
    }
    fclose(file);
    printf("Successfully saved matrix to %s\n", file_path);
}

Matrix* matrix_load(char* file_path) {
    FILE* file = fopen(file_path, "r");
    if (file == NULL) {
        fprintf(stderr, "MatrixLib error : cannot open '%s' for reading\n", file_path);
        exit(EXIT_FAILURE);
    }
    int rows, cols;
    if (fscanf(file, "%d %d", &rows, &cols) != 2) {
        fprintf(stderr, "MatrixLib error : invalid header in '%s'\n", file_path);
        exit(EXIT_FAILURE);
    }
    Matrix* m = matrix_create(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%lf", &m->entries[i][j]) != 1) {
                fprintf(stderr, "MatrixLib error : invalid data in '%s'\n", file_path);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);
    printf("Successfully loaded matrix from %s\n", file_path);
    return m;
}

#endif // _C_MATRIXLIB_