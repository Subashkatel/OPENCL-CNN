#ifndef CNN_H_
#define CNN_H_

// Standard library includes
#include <vector>
#include <unordered_map>
#include <assert.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <string>
#include <sys/time.h>
#include <cstring>
#include <sstream>
#include <OpenCL/opencl.h>

// Define image dimensions for each layer of the CNN
#define width_image_input_CNN   32  // Width of normalized input image
#define height_image_input_CNN  32  // Height of normalized input image
#define width_image_C1_CNN      28  // Width after first convolution layer
#define height_image_C1_CNN     28  // Height after first convolution layer
#define width_image_S2_CNN      14  // Width after first pooling layer
#define height_image_S2_CNN     14  // Height after first pooling layer
#define width_image_C3_CNN      10  // Width after second convolution layer
#define height_image_C3_CNN     10  // Height after second convolution layer
#define width_image_S4_CNN      5   // Width after second pooling layer
#define height_image_S4_CNN     5   // Height after second pooling layer
#define width_image_C5_CNN      1   // Width after third convolution layer
#define height_image_C5_CNN     1   // Height after third convolution layer
#define width_image_output_CNN  1   // Width of output layer
#define height_image_output_CNN 1   // Height of output layer

// Define kernel (filter) dimensions
#define width_kernel_conv_CNN    5  // Width of convolution kernel
#define height_kernel_conv_CNN   5  // Height of convolution kernel
#define width_kernel_pooling_CNN 2  // Width of pooling kernel
#define height_kernel_pooling_CNN 2 // Height of pooling kernel
#define size_pooling_CNN         2  // Size of pooling operation

// Define number of feature maps for each layer
#define num_map_input_CNN  1    // Number of input channels (1 for grayscale)
#define num_map_C1_CNN     6    // Number of feature maps after first convolution
#define num_map_S2_CNN     6    // Number of feature maps after first pooling
#define num_map_C3_CNN     16   // Number of feature maps after second convolution
#define num_map_S4_CNN     16   // Number of feature maps after second pooling
#define num_map_C5_CNN     120  // Number of feature maps after third convolution
#define num_map_output_CNN 10   // Number of output classes (digits 0-9)

// Define dataset sizes for MNIST
#define num_patterns_train_CNN 60000 // Number of training examples
#define num_patterns_test_CNN  10000 // Number of test examples

// Define training parameters
#define num_epochs_CNN     100   // Maximum number of training epochs
#define accuracy_rate_CNN  0.985 // Target accuracy rate
#define learning_rate_CNN  0.01  // Learning rate for gradient descent
#define eps_CNN            1e-8  // Small value to prevent division by zero

// Define lengths of weight and bias arrays for each layer
#define len_weight_C1_CNN     150   // 5*5*6*1 weights for first convolution layer
#define len_bias_C1_CNN       6     // 6 bias values for first convolution layer
#define len_weight_S2_CNN     6     // 6 weights for first pooling layer
#define len_bias_S2_CNN       6     // 6 bias values for first pooling layer
#define len_weight_C3_CNN     2400  // 5*5*16*6 weights for second convolution layer
#define len_bias_C3_CNN       16    // 16 bias values for second convolution layer
#define len_weight_S4_CNN     16    // 16 weights for second pooling layer
#define len_bias_S4_CNN       16    // 16 bias values for second pooling layer
#define len_weight_C5_CNN     48000 // 5*5*16*120 weights for third convolution layer
#define len_bias_C5_CNN       120   // 120 bias values for third convolution layer
#define len_weight_output_CNN 1200  // 120*10 weights for output layer
#define len_bias_output_CNN   10    // 10 bias values for output layer

// Define number of neurons in each layer
#define num_neuron_input_CNN  1024 // 32*32 input neurons
#define num_neuron_C1_CNN     4704 // 28*28*6 neurons after first convolution
#define num_neuron_S2_CNN     1176 // 14*14*6 neurons after first pooling
#define num_neuron_C3_CNN     1600 // 10*10*16 neurons after second convolution
#define num_neuron_S4_CNN     400  // 5*5*16 neurons after second pooling
#define num_neuron_C5_CNN     120  // 1*1*120 neurons after third convolution
#define num_neuron_output_CNN 10   // 10 output neurons (one per digit)

// Define indices for forward and backward propagation kernels
#define FORWARD_NUM  6
#define FORWARD_C1   0
#define FORWARD_S2   1
#define FORWARD_C3   2
#define FORWARD_S4   3
#define FORWARD_C5   4
#define FORWARD_OUT  5

#define BACKWARD_NUM 7
#define BACKWARD_OUT 0
#define BACKWARD_C5  1
#define BACKWARD_S4  2
#define BACKWARD_C3  3
#define BACKWARD_S2  4
#define BACKWARD_C1  5
#define BACKWARD_IN  6

class CNN {
public:
    CNN();
    ~CNN();

    void init(char* model=NULL);  // Initialize the CNN, optionally loading a pre-trained model
    bool train();                 // Train the CNN on the MNIST dataset
    int predict(const unsigned char *data, int width, int height);  // Predict a digit given an input image
    bool readModelFile(const char *name);  // Load a pre-trained model from a file
    bool saveMiddlePic(int index);  // Save intermediate layer outputs as images for visualization
    int init_opencl();  // Initialize OpenCL for GPU acceleration

protected:
    void release();  // Free allocated memory

    // Initialization functions
    bool initWeightThreshold();  // Initialize weights and biases with random values
    void init_variable(float* val, float c, int len);  // Initialize an array with a constant value
    bool uniform_rand(float* src, int len, float min, float max);  // Generate uniform random numbers
    float uniform_rand(float min, float max);  // Generate a single uniform random number

    // MNIST dataset handling functions
    int ReverseInt(int i);  // Helper function for reading MNIST file format
    void read_mnist_data(std::string filename, float* data_dst, int num_image);  // Read MNIST images
    void read_mnist_labels(std::string filename, float* data_dst, int num_image);  // Read MNIST labels
    bool get_src_data();  // Load MNIST dataset

    // Activation functions and their derivatives
    float activation_function_tanh(float x);
    float activation_function_tanh_derivative(float x);
    float activation_function_identity(float x);
    float activation_function_identity_derivative(float x);

    // Loss function and its derivative
    float loss_function_mse(float y, float t);  // Mean Squared Error
    float loss_function_mse_derivative(float y, float t);
    void loss_function_gradient(const float* y, const float* t, float* dst, int len);

    // Helper math functions
    float dot_product(const float* s1, const float* s2, int len);
    bool muladd(const float* src, float c, int len, float* dst);

    // Model saving function
    bool saveModelFile(const char* name);

    // Utility function for indexing 3D arrays
    int get_index(int x, int y, int channel, int width, int height, int depth);

    // Forward propagation functions
    bool Forward_C1(int index, cl_mem & Forward_in_mem0);
    bool Forward_S2();
    bool Forward_C3();
    bool Forward_S4();
    bool Forward_C5();
    bool Forward_output();

    // Backward propagation functions
    bool Backward_output(int index);
    bool Backward_C5();
    bool Backward_S4();
    bool Backward_C3();
    bool Backward_S2();
    bool Backward_C1();
    bool Backward_input(int index);

    // Weight update function
    bool UpdateWeights();
    void update_weights_bias(const float* delta, float* e_weight, float* weight, int len);

    // Evaluation function
    float test();  // Evaluate the model on the test set

    // Image saving function for debugging
    bool bmp8(const float *data, int width, int height, const char *name);

private:
    // Training and test data
    float* data_input_train;   // Input images for training
    float* data_output_train;  // Labels for training
    float* data_input_test;    // Input images for testing
    float* data_output_test;   // Labels for testing
    float* data_single_image;  // Single input image for prediction
    float* data_single_label;  // Single label for prediction

    // Weights and biases for each layer
    float weight_C1[len_weight_C1_CNN];
    float bias_C1[len_bias_C1_CNN];
    float weight_S2[len_weight_S2_CNN];
    float bias_S2[len_bias_S2_CNN];
    float weight_C3[len_weight_C3_CNN];
    float bias_C3[len_bias_C3_CNN];
    float weight_S4[len_weight_S4_CNN];
    float bias_S4[len_bias_S4_CNN];
    float weight_C5[len_weight_C5_CNN];
    float bias_C5[len_bias_C5_CNN];
    float weight_output[len_weight_output_CNN];
    float bias_output[len_bias_output_CNN];

    // Accumulated errors for weights and biases
    float E_weight_C1[len_weight_C1_CNN];
    float E_bias_C1[len_bias_C1_CNN];
    float E_weight_S2[len_weight_S2_CNN];
    float E_bias_S2[len_bias_S2_CNN];
    float E_weight_C3[len_weight_C3_CNN];
    float E_bias_C3[len_bias_C3_CNN];
    float E_weight_S4[len_weight_S4_CNN];
    float E_bias_S4[len_bias_S4_CNN];
    float* E_weight_C5;
    float* E_bias_C5;
    float* E_weight_output;
    float* E_bias_output;

    // Neuron activations for each layer
    float neuron_input[num_neuron_input_CNN];
    float neuron_C1[num_neuron_C1_CNN];
    float neuron_S2[num_neuron_S2_CNN];
    float neuron_C3[num_neuron_C3_CNN];
    float neuron_S4[num_neuron_S4_CNN];
    float neuron_C5[num_neuron_C5_CNN];
    float neuron_output[num_neuron_output_CNN];

    // Neuron deltas (errors) for each layer
    float delta_neuron_output[num_neuron_output_CNN];
    float delta_neuron_C5[num_neuron_C5_CNN];
    float delta_neuron_S4[num_neuron_S4_CNN];
    float delta_neuron_C3[num_neuron_C3_CNN];
    float delta_neuron_S2[num_neuron_S2_CNN];
    float delta_neuron_C1[num_neuron_C1_CNN];
    float delta_neuron_input[num_neuron_input_CNN];

    // Weight and bias deltas (errors) for each layer
    float delta_weight_C1[len_weight_C1_CNN];
    float delta_bias_C1[len_bias_C1_CNN];
    float delta_weight_S2[len_weight_S2_CNN];
    float delta_bias_S2[len_bias_S2_CNN];
    float delta_weight_C3[len_weight_C3_CNN];
    float delta_bias_C3[len_bias_C3_CNN];
    float delta_weight_S4[len_weight_S4_CNN];
    float delta_bias_S4[len_bias_S4_CNN];
    float delta_weight_C5[len_weight_C5_CNN];
    float delta_bias_C5[len_bias_C5_CNN];
    float delta_weight_output[len_weight_output_CNN];
    float delta_bias_output[len_bias_output_CNN];

    // OpenCL related variables
    cl_uint num_devs_returned;
    cl_context_properties properties[3];
    cl_device_id device_id;
    cl_int err;
    cl_int errs[FORWARD_NUM+1];
    cl_platform_id platform_id;
    cl_uint num_platforms_returned;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;

    // OpenCL kernels for forward and backward propagation
    cl_kernel Forward_kernel[FORWARD_NUM];
    cl_kernel Backward_kernel[BACKWARD_NUM];
    cl_kernel Update_weights;

    cl_kernel Backward_kernel_s2_weight;
    cl_kernel Backward_kernel_s2_bias;
    cl_kernel Backward_kernel_input_weight;
    cl_kernel Backward_kernel_input_bias;

    // OpenCL memory objects for input data
    cl_mem cl_data_input_train;
    cl_mem cl_label_input_train;
    cl_mem cl_data_input_test;
    cl_mem cl_label_input_test;

    // OpenCL memory objects for forward propagation
    cl_mem Forward_in_mem;
    cl_mem Forward_C1_mem;
    cl_mem Forward_S2_mem;
    cl_mem Forward_C3_mem;
    cl_mem Forward_S4_mem;
    cl_mem Forward_C5_mem;
    cl_mem Forward_out_mem;
    cl_mem Forward_bias[FORWARD_NUM];
    cl_mem Forward_weight[FORWARD_NUM];

    // OpenCL memory objects for backward propagation
    cl_mem Backward_bias[BACKWARD_NUM-1];
    cl_mem Backward_weight[BACKWARD_NUM-1];
    
    cl_mem Backward_out_mem;
    cl_mem Backward_C5_mem;
    cl_mem Backward_S4_mem;
    cl_mem Backward_C3_mem;
    cl_mem Backward_S2_mem;
    cl_mem Backward_C1_mem;
    cl_mem Backward_in_mem;

    // OpenCL memory objects for weight updates
    cl_mem Update_bias[FORWARD_NUM];
    cl_mem Update_weight[FORWARD_NUM];

    // Arrays to store memory sizes for forward and backward propagation
    const int for_mem_bw_len[FORWARD_NUM][2] = {
        {len_bias_C1_CNN, len_weight_C1_CNN},
        {len_bias_S2_CNN, len_weight_S2_CNN},
        {len_bias_C3_CNN, len_weight_C3_CNN},
        {len_bias_S4_CNN, len_weight_S4_CNN},
        {len_bias_C5_CNN, len_weight_C5_CNN},
        {len_bias_output_CNN, len_weight_output_CNN}
    };
    const int back_mem_bw_len[BACKWARD_NUM-1][2] = {
        {len_bias_output_CNN, len_weight_output_CNN},
        {len_bias_C5_CNN, len_weight_C5_CNN},
        {len_bias_S4_CNN, len_weight_S4_CNN},
        {len_bias_C3_CNN, len_weight_C3_CNN},
        {len_bias_S2_CNN, len_weight_S2_CNN},
        {len_bias_C1_CNN, len_weight_C1_CNN}
    };

    // Arrays to store neuron counts for each layer
    const int for_mem_in_out_len[FORWARD_NUM+1] = {
        num_neuron_input_CNN, num_neuron_C1_CNN, num_neuron_S2_CNN,
        num_neuron_C3_CNN, num_neuron_S4_CNN, num_neuron_C5_CNN, num_neuron_output_CNN
    };
    const int back_mem_in_out_len[BACKWARD_NUM] = {
        num_neuron_output_CNN, num_neuron_C5_CNN, num_neuron_S4_CNN,
        num_neuron_C3_CNN, num_neuron_S2_CNN, num_neuron_C1_CNN, num_neuron_input_CNN
    };

    // Pointers to OpenCL memory objects for each layer
    cl_mem *for_mem[FORWARD_NUM+1] = {
        &Forward_in_mem, &Forward_C1_mem, &Forward_S2_mem,
        &Forward_C3_mem, &Forward_S4_mem, &Forward_C5_mem, &Forward_out_mem
    };

    // Pointers to host memory for each layer
    float *for_mem_src[FORWARD_NUM+1] = {
        neuron_input, neuron_C1, neuron_S2,
        neuron_C3, neuron_S4, neuron_C5, neuron_output
    };

    // Pointers to OpenCL memory objects for backward propagation
    cl_mem *back_mem[BACKWARD_NUM] = {
        &Backward_out_mem, &Backward_C5_mem, &Backward_S4_mem,
        &Backward_C3_mem, &Backward_S2_mem, &Backward_C1_mem, &Backward_in_mem
    };

    // Pointers to host memory for backward propagation
    float *back_mem_src[BACKWARD_NUM] = {
        delta_neuron_output, delta_neuron_C5, delta_neuron_S4,
        delta_neuron_C3, delta_neuron_S2, delta_neuron_C1, delta_neuron_input
    };

    // Names of OpenCL kernels for forward propagation
    std::string forward_kernel_name[FORWARD_NUM] = {
        "kernel_forward_c1",
        "kernel_forward_s2",
        "kernel_forward_c3",
        "kernel_forward_s4",
        "kernel_forward_c5",
        "kernel_forward_output"
    };

    // Names of OpenCL kernels for backward propagation
    std::string backward_kernel_name[BACKWARD_NUM] = {
        "kernel_backward_output",
        "kernel_backward_c5",
        "kernel_backward_s4",
        "kernel_backward_c3",
        "kernel_backward_s2",
        "kernel_backward_c1",
        "kernel_backward_input"
    };

    // Pointers to bias arrays for each layer
    float *biases[FORWARD_NUM] = {
        bias_C1, bias_S2, bias_C3, bias_S4, bias_C5, bias_output
    };

    // Pointers to weight arrays for each layer
    float *weights[FORWARD_NUM] = {
        weight_C1, weight_S2, weight_C3, weight_S4, weight_C5, weight_output
    };
};

#endif /* CNN_H_ */