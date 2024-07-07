#include "cnn.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>

using namespace std;

#ifdef TEST_PREDICT
// Function to read a PNG file (simplified, assumes 8-bit grayscale)
std::vector<unsigned char> readPNG(const std::string& filename, int& width, int& height)
{
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return std::vector<unsigned char>();
    }

    // Skip PNG header and IHDR chunk
    file.seekg(16);
    
    // Read width and height
    file.read(reinterpret_cast<char*>(&width), 4);
    file.read(reinterpret_cast<char*>(&height), 4);
    
    // Convert from big-endian to host endian
    width = (width & 0xFF000000) >> 24 | (width & 0x00FF0000) >> 8 | (width & 0x0000FF00) << 8 | (width & 0x000000FF) << 24;
    height = (height & 0xFF000000) >> 24 | (width & 0x00FF0000) >> 8 | (width & 0x0000FF00) << 8 | (width & 0x000000FF) << 24;

    // Skip to the image data (this is a simplification, real PNG parsing is more complex)
    file.seekg(8, std::ios::cur);

    std::vector<unsigned char> data(width * height);
    file.read(reinterpret_cast<char*>(&data[0]), width * height);

    return data;
}

// Function to resize image (simple nearest neighbor)
std::vector<unsigned char> resizeImage(const std::vector<unsigned char>& input, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
{
    std::vector<unsigned char> output(outputWidth * outputHeight);
    float scaleX = static_cast<float>(inputWidth) / outputWidth;
    float scaleY = static_cast<float>(inputHeight) / outputHeight;

    for (int y = 0; y < outputHeight; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            int inputX = static_cast<int>(x * scaleX);
            int inputY = static_cast<int>(y * scaleY);
            output[y * outputWidth + x] = input[inputY * inputWidth + inputX];
        }
    }

    return output;
}

int test_CNN_predict(const char* model)
{
    CNN cnn2;
    bool flag = cnn2.readModelFile(model);
    if (!flag) {
        std::cout << "read cnn model error" << std::endl;
        return -1;
    }
    if(cnn2.init_opencl() == -1){
        cout << "error init opencl" << endl;
        return 1;
    }   
    
    int width = 32, height = 32;
    int target[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::string image_path = "images/";
    for (int i = 0; i < 10; ++i) {
        std::string str = image_path + static_cast<char>('0' + target[i]) + ".png";
        
        int imgWidth, imgHeight;
        std::vector<unsigned char> imgData = readPNG(str, imgWidth, imgHeight);
        if (imgData.empty()) {
            fprintf(stderr, "read image error: %s\n", str.c_str());
            return -1;
        }
        
        // Invert colors (255 - pixel value)
        for (size_t j = 0; j < imgData.size(); ++j) {
            imgData[j] = 255 - imgData[j];
        }
        
        // Resize image
        std::vector<unsigned char> resizedImg = resizeImage(imgData, imgWidth, imgHeight, width, height);
        
        int ret = cnn2.predict(&resizedImg[0], width, height);
        cnn2.saveMiddlePic(target[i]);
        fprintf(stdout, "the actual digit is: %d, correct digit is: %d\n", ret, target[i]);
    }
    return 0;
}
#endif

int main(int argc, char* argv[]) {
#ifdef TEST_PREDICT
    if(argc == 2)
        test_CNN_predict(argv[1]);
    else 
        test_CNN_predict("cnn.model");
#else
    CNN Tcnn;
    if(argc == 2)
        Tcnn.init(argv[1]);
    else 
        Tcnn.init();
    if(Tcnn.init_opencl() == -1){
        cout << "error init opencl" << endl;
        return 1;
    }
    Tcnn.train();
#endif
    cout << "!!!Hello ^_^ ** ^_^ ** ^_^!!!" << endl; // prints !!!Hello World!!!
    return 0;
}