MODEL = 4_GPU_opt

# Detect the operating system
UNAME_S := $(shell uname -s)

# Set default compiler and C++ standard
CC := clang++
CXXFLAGS := -O3 -arch arm64 -std=c++11 -Wall -Wno-unused-command-line-argument

# OpenCL flags
ifeq ($(UNAME_S),Darwin)
    # macOS (Apple)
    OPENCL = -framework OpenCL
    DIR_HEADER_APPLE_OPENCL = -I/System/Library/Frameworks/OpenCL.framework/Headers/
else ifeq ($(UNAME_S),Linux)
    # Check for NVIDIA GPU
    ifneq ($(shell which nvidia-smi),)
        OPENCL = -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lOpenCL
    else
        # Intel or other GPUs on Linux
        OPENCL = -I/usr/local/include -lOpenCL
    endif
else
    # Windows (assuming MSYS2 or similar)
    OPENCL = -I/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.0/include -L/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.0/lib/x64 -lOpenCL
endif

# Flags
FLAGS = $(OPENCL) $(DIR_HEADER_APPLE_OPENCL) -lm

SRC_FILES := $(wildcard *.cpp)
SRC_FILES := $(filter-out main.cpp,$(SRC_FILES))
OBJ_FILES := $(patsubst %.cpp, %.o, $(SRC_FILES))

.PHONY: all print train predict clean

all: print train predict

print:
	@echo "Source files: $(SRC_FILES)"
	@echo "Object files: $(OBJ_FILES)"

train: $(MODEL)

predict: $(MODEL)_Predict

$(MODEL): $(OBJ_FILES) main_train.o
	@echo "Building $(MODEL)..."
	@$(CC) $(CXXFLAGS) -o $@ $^ $(FLAGS)

$(MODEL)_Predict: $(OBJ_FILES) main_predict.o
	@echo "Building $(MODEL)_Predict..."
	@$(CC) $(CXXFLAGS) -o $@ $^ $(FLAGS)

%.o: %.cpp cnn.h
	@echo "Building $@..."
	@$(CC) $(CXXFLAGS) -c -o $@ $< $(FLAGS)

main_predict.o: main.cpp cnn.h
	@echo "Building $@..."
	@$(CC) $(CXXFLAGS) -c -o $@ $< -DTEST_PREDICT $(FLAGS)

main_train.o: main.cpp cnn.h
	@echo "Building $@..."
	@$(CC) $(CXXFLAGS) -c -o $@ $< $(FLAGS)

clean:
	@echo "Cleaning up..."
	@rm -f *.o $(MODEL) $(MODEL)_Predict