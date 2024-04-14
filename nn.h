#include <stdio.h>

// input    l1                  l2
//  ^
// a0 -> (a0*w1+b1) = a1 -> (a1+w2 + b2) = a2

typedef struct {
    size_t rows; 
    size_t cols; 
    size_t stride;
    float *es;
} Mat;

typedef struct {
    size_t layers; 
    Mat *ws;
    Mat *bs;
    Mat *as; // activation layers should one more layer than weight, biases
} NN;

typedef struct {
    NN nn; 
    Mat ti;
    Mat to;
} NN_Cost;


Mat mat_alloc(size_t rows, size_t cols);
Mat mat_row(Mat a, size_t idx);
void mat_cpy(Mat dst, Mat src);
void mat_dot(Mat a, Mat b, Mat dst);
void mat_sum(Mat a, Mat b);
void mat_relu(Mat a);
void mat_rand(Mat a);
void mat_fill(Mat a, size_t v);
void mat_print(Mat a, const char* name, int indent);


NN nn_alloc(size_t *layers, size_t layer_len);
float nn_cost(NN_Cost cost);
void nn_rand(NN nn);
void nn_forward(NN nn);
void nn_backprop(NN nn, Mat ti, Mat to);
void nn_print(NN nn, const char* name);
void nn_print_activs(NN nn, const char* name);
