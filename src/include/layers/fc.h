#ifndef OPENCL_FC_LAYER
#define OPENCL_FC_LAYER

/*
    Kernel descriptions for opencl fc layer go here
*/

// ---- OpenCL Kernel for Fully Connected Layer ----
__kernel void fc_forward(__global float* input, __global float* weights, __global float* output, int input_size, int output_size) {
    int i = get_global_id(0);
    float sum = 0.0f;
    for (int j = 0; j < input_size; j++) {
        sum += input[j] * weights[j * output_size + i];
    }
    output[i] = sum;
}

#endif