CC := g++ -march=native
OPENCL = -I/usr/local/include -L/usr/local/cuda/lib64 -lOpenCL 
OPENCV = $(shell pkg-config opencv --cflags --libs)