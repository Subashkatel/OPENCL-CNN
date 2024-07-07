/*
 * MNIST Dataset Reader and Parser
 * 
 * This file contains functions to read and parse the MNIST dataset,
 * which is a large database of handwritten digits commonly used
 * for training various image processing systems.
 */

#include "./cnn.h"
#include <cassert>
#include <fstream>
#include <iso646.h>
#include <istream>

// Reverse the byte order of a 32-bit integer
// This is necessary because MNIST files are in big-endian format
int CNN::ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;         // Extract the least significant byte
    ch2 = (i >> 8) & 255;  // Extract the second byte
    ch3 = (i >> 16) & 255; // Extract the third byte
    ch4 = (i >> 24) & 255; // Extract the most significant byte
    // Combine the bytes in reverse order
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

// Read MNIST image data from a file and store it in the provided array
void CNN::read_mnist_data(std::string filename, float *data, int num_images) {
    std::ifstream file(filename, std::ios::binary);
    const int src_image_width = 28;  // Original MNIST image width
    const int src_image_height = 28; // Original MNIST image height
    const int padding_x = 2;         // Horizontal padding to be added
    const int padding_y = 2;         // Vertical padding to be added
    const float scale_min = -1;      // Minimum value for scaling
    const float scale_max = 1;       // Maximum value for scaling
    
    if(file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int num_rows = 0;
        int num_cols = 0;
        
        // Read and verify the magic number
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        
        // Read and verify the number of images
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        assert(number_of_images == num_images);
        
        // Read and verify the number of rows
        file.read((char*)&num_rows, sizeof(num_rows));
        num_rows = ReverseInt(num_rows);
        assert(num_rows == src_image_height);
        
        // Read and verify the number of columns
        file.read((char*)&num_cols, sizeof(num_cols));
        num_cols = ReverseInt(num_cols);
        assert(num_cols == src_image_width);
        
        int single_image_size = height_image_input_CNN * width_image_input_CNN;
        
        // Read image data
        for (int i = 0; i < number_of_images; ++i) {
            int address_offset = single_image_size * i;
            for (int r = 0; r < num_rows; ++r) {
                for (int c = 0; c < num_cols; ++c) {
                    unsigned char temp_value = 0;
                    file.read((char*)&temp_value, sizeof(temp_value));
                    // Scale and store the pixel value
                    int idx = address_offset + width_image_input_CNN * (r + padding_y) + (c + padding_x);
                    data[idx] = (temp_value / 255.0) * (scale_max - scale_min) + scale_min;
                }
            }
        }
    }   
}

// Read MNIST label data from a file and store it in the provided array
void CNN::read_mnist_labels(std::string filename, float* data_dst, int num_image)
{
	const float scale_max = 0.8;

	std::ifstream file(filename, std::ios::binary);
	assert(file.is_open());

	int magic_number = 0;
	int number_of_images = 0;
	file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = ReverseInt(magic_number);
	file.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = ReverseInt(number_of_images);
	assert(number_of_images == num_image);

	for (int i = 0; i < number_of_images; ++i) {
		unsigned char temp = 0;
		file.read((char*)&temp, sizeof(temp));
		data_dst[i * num_map_output_CNN + temp] = scale_max;
	}
}

// Load both training and test data from MNIST dataset files
bool CNN::get_src_data() {
    assert(data_input_train && data_output_train && data_input_test && data_output_test);
    std::cout << "Reading MNIST dataset" << std::endl;

    // Read training data
    std::string filename_train_images = "data/train-images-idx3-ubyte";
    std::string filename_train_labels = "data/train-labels-idx1-ubyte";
    read_mnist_data(filename_train_images, data_input_train, num_patterns_train_CNN);
    read_mnist_labels(filename_train_labels, data_output_train, num_patterns_train_CNN);

    // Read test data
    std::string filename_test_images = "data/t10k-images-idx3-ubyte";
    std::string filename_test_labels = "data/t10k-labels-idx1-ubyte";
    read_mnist_data(filename_test_images, data_input_test, num_patterns_test_CNN);
    read_mnist_labels(filename_test_labels, data_output_test, num_patterns_test_CNN);

    return true;
}