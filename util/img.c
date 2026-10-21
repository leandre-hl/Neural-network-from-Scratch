#include "../parameters.h"
#include "./img.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Img** csv_to_imgs(char* file_path_str, int start, int number_of_imgs) {
    // Extracts the images from the file
    // Returns an array of Image structs
    // Image->img_data is a column vector of the size 784x1

    if (start < 0 || number_of_imgs <= 0) {
        fprintf(stderr, "Invalid CSV image range\n");
        exit(EXIT_FAILURE);
    }

    char row[MAX_MNIST_CHAR];
    Img** imgs = malloc(number_of_imgs * sizeof(Img*));
    if (imgs == NULL) {
        fprintf(stderr, "Out of memory while loading images\n");
        exit(EXIT_FAILURE);
    }
    FILE* file = fopen(file_path_str, "r");
    if (file == NULL) {
        perror(file_path_str);
        free(imgs);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i <= start; i++) {
        if (fgets(row, sizeof(row), file) == NULL) {
            fprintf(stderr, "CSV does not contain the requested starting row\n");
            fclose(file);
            free(imgs);
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < number_of_imgs; i++) {
        if (fgets(row, sizeof(row), file) == NULL) {
            fprintf(stderr, "CSV ended before %d images could be loaded\n", number_of_imgs);
            for (int j = 0; j < i; j++) img_free(imgs[j]);
            fclose(file);
            free(imgs);
            exit(EXIT_FAILURE);
        }

        Img* img = malloc(sizeof(Img));
        if (img == NULL) {
            fprintf(stderr, "Out of memory while loading images\n");
            for (int j = 0; j < i; j++) img_free(imgs[j]);
            fclose(file);
            free(imgs);
            exit(EXIT_FAILURE);
        }
        img->img_data = matrix_create(IMG_PIXELS, 1);
        char* token = strtok(row, ",");
        int field = 0;
        while (token != NULL) {
            if (field == 0) {
                img->label = atoi(token);
            } else if (field <= IMG_PIXELS) {
                img->img_data->entries[field - 1][0] = atof(token) / 255.0;
            } else {
                fprintf(stderr, "CSV row contains too many fields\n");
                matrix_free(img->img_data);
                free(img);
                for (int j = 0; j < i; j++) img_free(imgs[j]);
                fclose(file);
                free(imgs);
                exit(EXIT_FAILURE);
            }
            field++;
            token = strtok(NULL, ",");
        }
        if (field != IMG_PIXELS + 1) {
            fprintf(stderr, "CSV row does not contain a label and %d pixels\n", IMG_PIXELS);
            matrix_free(img->img_data);
            free(img);
            for (int j = 0; j < i; j++) img_free(imgs[j]);
            fclose(file);
            free(imgs);
            exit(EXIT_FAILURE);
        }
        imgs[i] = img;
    }
    fclose(file);
    printf("Successfully loaded %d images from %s\n", number_of_imgs, file_path_str);
    return imgs;
}

void img_free(Img* img) {
	matrix_free(img->img_data);
	free(img);
	img = NULL;
}

void imgs_free(Img** imgs, int n) {
	for (int i = 0; i < n; i++) {
		img_free(imgs[i]);
	}
	free(imgs);
	imgs = NULL;
}

void img_print(Img* img, int rows, int cols) {
    // Img->img_data must be a column vector
    // Img->img_data size must be equal to rows * cols
    printf("============================================================\n");

    if (img->img_data->cols != 1) {
        printf("Error : expected a column matrix in img_print, got %dx%d", img->img_data->rows, img->img_data->cols);
        exit(1);
    }
    if (img->img_data->rows != rows * cols) {
        printf("Error : %dx%d img in img_print can't fit in a %dx%d matrix", img->img_data->rows, img->img_data->cols, rows, cols);
        exit(1);
    }
    

    for (int i = 0; i < rows; i++) {
        printf("||");
        for (int j = 0; j < cols; j++) {
            if (img->img_data->entries[i*cols + j][0] == 0.0) {
                // No pixel
                printf("  ");
            }
            else {
                // Pixel
                printf("O ");
            }
        }
        printf("||\n");

    }    
    printf("Label: %d\n", img->label);
    printf("============================================================\n");

}