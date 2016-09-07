/*! \file searchInTheBoxCuda.cu
*CUDA file with search methods
*/

#define MAX_SIZE 40
#define NUM_DIMENSIONS 3
#define MAX_RESULT_SIZE 512
#define RANGE 0.2f;
#define BLOCKSIZE 256
/*! Number of threads per thread group that are working in the same query*/
#define THREADS_PER_QUERY 4
/*! NUmber of queries per block. Derived by BLOCKSIZE and THREADS_PER_QUERY*/
#define QUERIES_PER_BLOCK (BLOCKSIZE/THREADS_PER_QUERY)
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

/*! CUDA stack that uses atomic operations*/
typedef struct
{    
    unsigned int data[MAX_SIZE];
    int size;
} Queue;


/*! Atomic stack push*/
__device__ bool push_back(Queue* queue, unsigned int index)
{
    int idx = atomicAdd(&queue->size,1);
    if (idx < MAX_SIZE)
    {
	queue->data[idx] = index;
        return true;
    }
    atomicSub(&queue->size,1);
    return false;
    
}

/*! Atomic stack pop*/
__device__ unsigned int pop_front(Queue* queue)
{
   int idx = atomicSub(&queue->size,1)-1;
    if (idx >= 0)
    {
	unsigned int element = queue->data[idx];
        return element;
    }
    atomicAdd(&queue->size,1);
    return INT_MAX;
}


__device__ unsigned int leftSonIndex(unsigned int index)
{
    return 2 * index + 1;
}


__device__ unsigned int rightSonIndex(unsigned int index)
{
    return 2 * index + 2;
}


__device__ bool intersects(unsigned int index,  float* theDimensions, unsigned int nPoints,
                           float* minPoint, float* maxPoint, int dimension)
{
    return (theDimensions[nPoints * dimension + index] <= maxPoint[dimension]
            && theDimensions[nPoints * dimension + index] >= minPoint[dimension]);
}


__device__ bool isInTheBox(unsigned int index,  float* theDimensions, unsigned int nPoints,
                           float* minPoint, float* maxPoint)
{
    bool inTheBox = true;
    for (int i = 0; i < NUM_DIMENSIONS; ++i)
    {
        inTheBox &= (theDimensions[nPoints * i + index] <= maxPoint[i]
                     && theDimensions[nPoints * i + index] >= minPoint[i]);
    }
    
    return inTheBox;
}

//! The k-d tree search function that uses a shared stack
/*! This k-d tree search __global__ function performs box queries in the k-d tree. The k-d tree points are used for box centers. 
* CUDA threads are group together and each group shares a search stack (DFS is used). Synchronization through atomics. THREADS_PER_QUERY, QUERIES_PER_BLOCK, BLOCKSIZE defined * in the file.
\param nPoints the number of points in the k-d tree
\param dimensions the coordinates of the points in the k-d tree
\param ids the identifiers of the points in the k-d tree
\param results the array with the results of the queries
*/
__global__ void CUDASearchInTheKDBox(unsigned int nPoints,  float* dimensions,  unsigned int* ids,  unsigned int* results)
{
	unsigned int point_index = blockIdx.x*QUERIES_PER_BLOCK+threadIdx.x/THREADS_PER_QUERY;
	unsigned int tid = threadIdx.x%THREADS_PER_QUERY;
	unsigned int gid = threadIdx.x/THREADS_PER_QUERY;
	/*shared structure*/
	__shared__ Queue indecesToVisit[QUERIES_PER_BLOCK];
	__shared__ unsigned int pointsFound[QUERIES_PER_BLOCK];
	
	
	if(point_index < nPoints) {
		/*group gets its respective box*/
		int theDepth = floor(log2((float)nPoints));
		float minPoint[NUM_DIMENSIONS];
		float maxPoint[NUM_DIMENSIONS];
		for(int i = 0; i<NUM_DIMENSIONS; ++i) {
			minPoint[i] = dimensions[nPoints*i+point_index] - RANGE;
			maxPoint[i] = dimensions[nPoints*i+point_index] + RANGE;
		}

		unsigned int resultIndex = nPoints + MAX_RESULT_SIZE*point_index;
		if (tid == 0) {
			indecesToVisit[gid].size =0;
			push_back(&indecesToVisit[gid], 0);
			pointsFound[gid] = 0;
		}

		__syncthreads();
        	/*terminate when stack is empty-syncthreads ensure that this check is synchronized*/
        	while (indecesToVisit[gid].size > 0) {
			__syncthreads();
			/*each thread gets its index*/
			unsigned int index = pop_front(&indecesToVisit[gid]);	
			int dimension = ((int) floor(log2((float)index+1))) % NUM_DIMENSIONS;
			/*expand endex*/
			if (index < nPoints) {
				bool intersection = intersects(index,dimensions, nPoints, minPoint, maxPoint,dimension);

				if(intersection && isInTheBox(index, dimensions, nPoints, minPoint, maxPoint)) {
					if(pointsFound[gid] < MAX_RESULT_SIZE) {
						int offset = atomicAdd(&pointsFound[gid], 1);
						results[resultIndex+offset] = index;
					}
				}
			
				bool isLowerThanBoxMin = dimensions[nPoints*dimension + index] < minPoint[dimension];
	                	int startSon = isLowerThanBoxMin;

        	        	int endSon = isLowerThanBoxMin || intersection;
		
				for (int whichSon = startSon; whichSon < endSon + 1; ++whichSon) {
					unsigned int indexToAdd = leftSonIndex(index) + whichSon;

					if (indexToAdd < nPoints) {
						push_back(&indecesToVisit[gid],indexToAdd);
					}
				}
			}
			/*synchronoze for next check*/
			__syncthreads();
		}
		
		/*post results*/
		if (tid == 0) {
			results[point_index] = pointsFound[gid];
		}
	}    
}

void CUDAKernelWrapper(unsigned int nPoints,float *d_dim,unsigned int *d_ids,unsigned int *d_results)
{


    // Number of thread blocks
	unsigned int gridSize = (int)ceil((float)nPoints*THREADS_PER_QUERY/BLOCKSIZE);

	CUDASearchInTheKDBox<<<gridSize, BLOCKSIZE>>>(nPoints, d_dim, d_ids,d_results);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));    
}

