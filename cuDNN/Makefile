CUDA_PATH ?= /usr/local/cuda
CUDNN_PATH ?= /home/sriram/anaconda3/envs/adabits

INCLUDES = -I$(CUDA_PATH)/include -I$(CUDNN_PATH)/include
LIBRARIES = -L$(CUDA_PATH)/lib64 -L$(CUDNN_PATH)/lib

all: seer

seer:
	nvcc collect_with_algo.cu $(INCLUDES) $(LIBRARIES) -lcudnn -lcublas -std=c++11 -arch=compute_61 -code=sm_61 -o collect_with_algo
	nvcc collect_without_algo.cu $(INCLUDES) $(LIBRARIES) -lcudnn -lcublas -std=c++11 -arch=compute_61 -code=sm_61 -o collect_without_algo

clean:
	rm -f collect_with_algo collect_without_algo