#ifndef _H_IMG_
#define _H_IMG_

#include "../MatrixLib/MatrixLib.h"

typedef struct {
    Matrix* img_data;
    int label;
} Img;

Img** csv_to_imgs(char* file_path_str, int start, int number_of_imgs);
void imgs_free(Img** imgs, int n);
void img_free(Img* img);
void img_print(Img* img, int rows, int cols);

#endif // _H_IMG_