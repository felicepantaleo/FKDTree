CFLAGS= -std=c++11 -ftree-vectorize -march=native -ffast-math -Ofast -mtune=native 
LIBS = -ltbb

CUDAOBJECT =
USE_CUDA = NO
USE_OPENCL = YES

CC=g++
NVCC=nvcc
###CUDA TEST
NVCC_RESULT := $(shell which nvcc 2> NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))


ifeq ($(USE_CUDA),YES)

ifeq ($(NVCC_TEST),nvcc)

CUDALIBS = -lcuda -lcudart -L/usr/local/cuda/lib64
CUDAFLAGS = -std=c++11 -D__USE_CUDA__
CUDACFLAGS = "-std=c++11  -O3" 
CUDAHEADERS = -I/usr/local/cuda/include

CUDAOBJECT = searchInTheBoxCuda.o

else

all: 
	$(error "Unable to provide CUDA support. NVCC not found.")

endif
endif

###OPENCL TEST
ifeq ($(USE_OPENCL),YES)
OPENCLLIBS = -lOpenCL
OPENCLFLAGS = -D__USE_OPENCL__
OPENCLHEADERS =

endif

all: kdtree

searchInTheBoxCuda.o: searchInTheBoxCuda.cu
	$(NVCC) $(CUDAHEADERS) $(CUDAFLAGS) -c -Xcompiler $(CUDACFLAGS) searchInTheBoxCuda.cu

kdtree: $(CUDAOBJECT)
	$(CC) main.cpp $(CFLAGS) $(CUDAFLAGS) $(OPENCLFLAGS) -o kdtree $(CUDAHEADERS) $(CUDALIBS) $(LIBS) $(OPENCLLIBS) $(CUDAOBJECT)

clean:
	rm -rf *.o kdtree



