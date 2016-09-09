LIBS = -ltbb

NVCC=nvcc
NVCC_RESULT := $(shell which nvcc 2> NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))

CUDALIBS = -lcuda -lcudart -L/usr/local/cuda/lib64
CUDAFLAGS = -g -std=c++11 -D_FORCE_INLINES
CUDAHEADERS = -I/usr/local/cuda/include




all: kdtree

kdtree: test_build.cu
	$(NVCC) $(CUDAHEADERS) $(CUDAFLAGS) $(CUDACFLAGS) -o kdtree test_build.cu $(LIBS)

clean:
	rm -rf kdtree



