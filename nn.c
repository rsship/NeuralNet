#include "nn.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAT_INDEX(a, i, j) (a).es[(a).stride*(i)+(j)]
#define ARRAY_LEN(a) sizeof((a))/sizeof((a)[0])

#define NN_INPUT(m) (m).as[0]
#define NN_OUTPUT(m) (m).as[(m).layers]

float rand_float() 
{
     return (float)rand() / (float)RAND_MAX; 
}

Mat mat_alloc(size_t rows, size_t cols) 
{
    Mat mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.stride = cols;
    mat.es = malloc(sizeof(*mat.es)*rows*cols);
    assert(mat.es != NULL);
    return mat;
}

Mat mat_row(Mat a, size_t idx)
{
    return (Mat){
        .rows = 1,
        .cols = a.cols,
        .stride = a.stride,
        .es = &MAT_INDEX(a, idx, 0),
    };
}

// NOTE; find a better way to cpy sub matrixs

void mat_cpy(Mat dst, Mat src)
{
    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.stride; ++j) {
            MAT_INDEX(dst, i, j) = MAT_INDEX(src, i, j);
        }
    }
}


// [2,2] X [2, 1] -> [2, 1]
void mat_dot(Mat a, Mat b, Mat dst)
{

    assert(a.cols == b.rows);
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);
    size_t n = a.cols;

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            for (size_t k = 0; k < n; ++k) {
                MAT_INDEX(dst, i , j) += MAT_INDEX(a, i, k) * MAT_INDEX(b, j, k);
            }
        }
    }
}

void mat_sum(Mat a, Mat b)
{
    assert(a.rows == b.rows);
    assert(a.cols == b.cols);

    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            MAT_INDEX(a, i, j) += MAT_INDEX(a, i, j);
        }
    }
}

void mat_relu(Mat a)
{
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            MAT_INDEX(a, i, j) = fmaxf(0, MAT_INDEX(a, i, j));
        }
    }
}


void mat_rand(Mat a)
{
    for(size_t i = 0; i < a.rows; ++i) {
        for(size_t j = 0; j < a.cols; ++j) {
            MAT_INDEX(a, i, j) = rand_float();
        }
    }
}

void mat_fill(Mat a, size_t v)
{
    for(size_t i = 0; i < a.rows; ++i) {
        for(size_t j = 0; j < a.cols; ++j) {
            MAT_INDEX(a, i, j) = v;
        }
    }
}

void mat_print(Mat a, const char* name, int indent)
{
    printf("%*s%s => [\n", indent, "",  name);
    for(size_t i = 0; i < a.rows; ++i) {
        for(size_t j = 0; j < a.cols; ++j) {
            printf("%*s%f", 2*indent, "", MAT_INDEX(a, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", indent, "");
}
/// [2, 2, 1]
NN nn_alloc(size_t *layers, size_t layer_len)
{
    NN nn;
    nn.layers = layer_len-1;

    nn.as = malloc(sizeof(*nn.as)*layer_len);
    assert(nn.as != NULL);

    nn.ws = malloc(sizeof(*nn.ws)*nn.layers);
    assert(nn.ws != NULL);
    
    nn.bs = malloc(sizeof(*nn.bs)*nn.layers);
    assert(nn.bs != NULL);

    nn.as[0] = mat_alloc(1, layers[0]);

    for(size_t i = 1; i < layer_len; ++i) {
        nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, layers[i]);
        nn.as[i] = mat_alloc(nn.as[i-1].rows, nn.ws[i-1].cols);
        nn.bs[i-1] = mat_alloc(nn.as[i-1].rows, nn.ws[i-1].cols);
    }
    return nn;
}

void nn_rand(NN nn)
{
    for (size_t i = 0; i < nn.layers; ++i) {
        mat_rand(nn.ws[i]);
        mat_rand(nn.bs[i]);
    }
}

void nn_print(NN nn, const char* name)
{
    char buffer[256];
    printf("%s -> [\n", name);
    for (size_t i = 0; i < nn.layers; ++i) {
        snprintf(buffer, 4, "ws%zu", i); 
        mat_print(nn.ws[i], buffer, 4);
        snprintf(buffer, 4, "bs%zu", i);
        mat_print(nn.bs[i], buffer, 4);
    }
    printf("]\n");
}


void nn_print_activs(NN nn, const char* name)
{
    char buffer[256];
    printf("%s -> [\n", name);
    for (size_t i = 0; i < nn.layers; ++i) {
        snprintf(buffer, 4, "as%zu", i); 
        mat_print(nn.as[i], buffer, 4);
    }
    printf("]");

}

void nn_forward(NN nn) 
{
    for (size_t i = 1; i <= nn.layers; ++i) {
        mat_dot(nn.as[i-1], nn.ws[i-1], nn.as[i]);
        mat_sum(nn.as[i], nn.bs[i-1]);
        mat_relu(nn.as[i]);
    }
}

float nn_cost(NN_Cost cost)
{
    size_t N = cost.ti.rows;
    float C = 0;
    for (size_t i = 0; i < cost.ti.rows; ++i) {
        Mat x = mat_row(cost.ti, i);
        Mat y = mat_row(cost.to, i);

        mat_cpy(NN_INPUT(cost.nn), x);
        nn_forward(cost.nn);
        
        for (size_t j = 0; j < cost.to.cols; ++j) {
            float d = MAT_INDEX(NN_OUTPUT(cost.nn), 0, j) - MAT_INDEX(y, 0, j);
            C += d*d;
        }
    }
    return C/N;
}

void nn_backprop(NN nn, Mat ti, Mat to)
{

    // forward NN
    // back propagte until input;

    for (size_t i = 0; i < ti.rows; ++i) {
        mat_cpy(NN_INPUT(nn), mat_row(ti, i));
        nn_forward(nn);
        // NN_OUTPUT(nn) == mat_row(to, i);
    }


}

float data[] = {
    0, 0, 0, 
    0, 1, 1, 
    1, 0, 1,
    1, 1, 1,
};

int main() 
{
    srand(time(0));
    
    Mat ti = {.rows = 4, .cols = 2, .stride = 3, .es = data};
    Mat to = {.rows = 4, .cols = 1, .stride = 3, .es = &data[2]};

    size_t layers[] = {2, 2, 1};
    NN nn = nn_alloc(layers, ARRAY_LEN(layers));

    nn_rand(nn);
    NN_Cost cost = {nn, ti, to};
    printf("%f\n", nn_cost(cost));

    return 0;
}
