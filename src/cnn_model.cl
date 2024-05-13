// ... code

#include <layers/conv.h>
#include <layers/fc.h>
#include <layers/pooling.h>

#define WIDTH 32
#define HEIGHT 32
#define DEPTH 3
#define FILTER_SIZE 5
#define POOL_SIZE 2
#define FC_SIZE 128

__kernel void cnn(__global float* input, __global float* output, __global float* filters, __global float* fcWeights, __global float* fcBiases) {
    __global float* convOutput1 = malloc(WIDTH * HEIGHT * DEPTH * sizeof(float));
    __global float* poolOutput1 = malloc(WIDTH/POOL_SIZE * HEIGHT/POOL_SIZE * DEPTH * sizeof(float));
    __global float* convOutput2 = malloc(WIDTH/POOL_SIZE * HEIGHT/POOL_SIZE * DEPTH * sizeof(float));
    __global float* poolOutput2 = malloc(WIDTH/(POOL_SIZE*POOL_SIZE) * HEIGHT/(POOL_SIZE*POOL_SIZE) * DEPTH * sizeof(float));
    __global float* convOutput3 = malloc(WIDTH/(POOL_SIZE*POOL_SIZE) * HEIGHT/(POOL_SIZE*POOL_SIZE) * DEPTH * sizeof(float));
    __global float* poolOutput3 = malloc(WIDTH/(POOL_SIZE*POOL_SIZE*POOL_SIZE) * HEIGHT/(POOL_SIZE*POOL_SIZE*POOL_SIZE) * DEPTH * sizeof(float));
    __global float* convOutput4 = malloc(WIDTH/(POOL_SIZE*POOL_SIZE*POOL_SIZE) * HEIGHT/(POOL_SIZE*POOL_SIZE*POOL_SIZE) * DEPTH * sizeof(float));
    __global float* poolOutput4 = malloc(WIDTH/(POOL_SIZE*POOL_SIZE*POOL_SIZE*POOL_SIZE) * HEIGHT/(POOL_SIZE*POOL_SIZE*POOL_SIZE*POOL_SIZE) * DEPTH * sizeof(float));
    __global float* fcOutput1 = malloc(FC_SIZE * sizeof(float));
    __global float* fcOutput2 = malloc(FC_SIZE * sizeof(float));

    conv2D(input, convOutput1, filters, WIDTH, HEIGHT, FILTER_SIZE);
    maxPooling(convOutput1, poolOutput1, WIDTH, HEIGHT, DEPTH, POOL_SIZE);

    conv2D(poolOutput1, convOutput2, filters + FILTER_SIZE * FILTER_SIZE, WIDTH/POOL_SIZE, HEIGHT/POOL_SIZE, FILTER_SIZE);
    maxPooling(convOutput2, poolOutput2, WIDTH/POOL_SIZE, HEIGHT/POOL_SIZE, DEPTH, POOL_SIZE);

    conv2D(poolOutput2, convOutput3, filters + 2 * FILTER_SIZE * FILTER_SIZE, WIDTH/(POOL_SIZE*POOL_SIZE), HEIGHT/(POOL_SIZE*POOL_SIZE), FILTER_SIZE);
    maxPooling(convOutput3, poolOutput3, WIDTH/(POOL_SIZE*POOL_SIZE), HEIGHT/(POOL_SIZE*POOL_SIZE), DEPTH, POOL_SIZE);

    conv2D(poolOutput3, convOutput4, filters + 3 * FILTER_SIZE * FILTER_SIZE, WIDTH/(POOL_SIZE*POOL_SIZE*POOL_SIZE), HEIGHT/(POOL_SIZE*POOL_SIZE*POOL_SIZE), FILTER_SIZE);
    maxPooling(convOutput4, poolOutput4, WIDTH/(POOL_SIZE*POOL_SIZE*POOL_SIZE), HEIGHT/(POOL_SIZE*POOL_SIZE*POOL_SIZE), DEPTH, POOL_SIZE);

    fullyConnected(poolOutput4, fcOutput1, fcWeights, fcBiases, WIDTH/(POOL_SIZE*POOL_SIZE*POOL_SIZE*POOL_SIZE) * HEIGHT/(POOL_SIZE*POOL_SIZE*POOL_SIZE*POOL_SIZE) * DEPTH, FC_SIZE);
    fullyConnected(fcOutput1, fcOutput2, fcWeights + FC_SIZE, fcBiases + FC_SIZE, FC_SIZE, FC_SIZE);

    for(int i = 0; i < FC_SIZE; i++) {
        output[i] = fcOutput2[i];
    }

    free(convOutput1);
    free(poolOutput1);
    free(convOutput2);
    free(poolOutput2);
    free(convOutput3);
    free(poolOutput3);
    free(convOutput4);
    free(poolOutput4);
    free(fcOutput1);
    free(fcOutput2);
}

// ... code