/*! Number of trees in the forest*/
#define NTREES 2
#define MAX_SIZE 128
#define NUM_DIMENSIONS 3
#define MAX_RESULT_SIZE 512
#define RANGE 0.2f;
#define BLOCKSIZE 256
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>

/*! CUDA queue, not thread safe*/
typedef struct
{
    
    unsigned int data[MAX_SIZE];
    unsigned int front;
    unsigned int tail;
    unsigned int size;
} Queue;

/*! Insert node index in queue*/
__device__ bool push_back(Queue* queue, unsigned int index)
{
    if (queue->size < MAX_SIZE)
    {
        queue->data[queue->tail] = index;
        queue->tail = (queue->tail + 1) % MAX_SIZE;
        queue->size++;
        return true;
    }
    return false;
    
}

/*! Extract node index from queue*/
__device__ unsigned int pop_front(Queue* queue)
{
    if (queue->size > 0)
    {
        unsigned int element = queue->data[queue->front];
        queue->front = (queue->front + 1) % MAX_SIZE;
        queue->size--;
        return element;
    }
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


//! The k-d tree search function over a forest of k-d trees
/*! This k-d tree search __global__ function performs box queries in the k-d tree forest. The points are used for box centers. 
* CUDA threads are group together and each group is assigned a box. Each thread searches through a different forest. 
* NTREES, BLOCKSIZE defined * in the file.
\param nPoints the number of points in the k-d tree
\param dimensions the coordinates of the points in the k-d tree
\param ids the identifiers of the points in the k-d tree
\param results the array with the results of the queries
*/
__global__ void CUDASearchInTheKDBox(unsigned int nPoints,  float* dimensions,  unsigned int* ids,  unsigned int* results)
{
    
    // Global Thread ID
	unsigned int point_index = blockIdx.x*(blockDim.x/NTREES)+threadIdx.x/NTREES;
	unsigned int thread_index = threadIdx.x%NTREES;    
	unsigned int group_index = threadIdx.x/NTREES;
	unsigned int unit = 1;

	__shared__ unsigned int pointsFound[BLOCKSIZE/NTREES];

	if(point_index < nPoints) {
		/*test part: locate corresponding point in the forest to get the box*/
		unsigned int step = (unsigned int) ceil((float) nPoints/NTREES);
		unsigned int forest_index = point_index/step;
		unsigned int inner_index = point_index%step;

		unsigned int offset = forest_index*step;

		unsigned int tPoints = (offset+step > nPoints)? nPoints-offset:step;
        
		int theDepth = floor(log2((float)tPoints));
		float minPoint[NUM_DIMENSIONS];
		float maxPoint[NUM_DIMENSIONS];
		for(int i = 0; i<NUM_DIMENSIONS; ++i) {
			minPoint[i] = dimensions[3*offset+tPoints*i+inner_index] - RANGE;
			maxPoint[i] = dimensions[3*offset+tPoints*i+inner_index] + RANGE;
		}
		/*identify the target tree for search*/
		offset = thread_index*step;
		tPoints = (offset+step > nPoints)? nPoints-offset:step;        
		ids += offset;
		dimensions += offset*NUM_DIMENSIONS;

		Queue indecesToVisit;
		indecesToVisit.front = indecesToVisit.tail =indecesToVisit.size =0;
		if (thread_index == 0) 
			pointsFound[group_index] = 0;
		__syncthreads ();

		unsigned int resultIndex = nPoints + MAX_RESULT_SIZE*point_index;
		push_back(&indecesToVisit, 0);
		/*a bfs for each tree*/
		for (int depth = 0; depth < theDepth + 1; ++depth) {
			int dimension = depth % NUM_DIMENSIONS;
			unsigned int numberOfIndecesToVisitThisDepth = indecesToVisit.size;
			for (unsigned int visitedIndecesThisDepth = 0; visitedIndecesThisDepth < numberOfIndecesToVisitThisDepth; visitedIndecesThisDepth++) {
				unsigned int index = pop_front(&indecesToVisit);
		
				bool intersection = intersects(index,dimensions, tPoints, minPoint, maxPoint, dimension);
                
                		if(intersection && isInTheBox(index, dimensions, tPoints, minPoint, maxPoint)) {
					unsigned int offset = atomicAdd(&pointsFound[group_index],unit);
					if (offset < MAX_RESULT_SIZE)
						results[resultIndex+offset] = index;
				}
                
				bool isLowerThanBoxMin = dimensions[tPoints*dimension + index] < minPoint[dimension];
				int startSon = isLowerThanBoxMin;
				int endSon = isLowerThanBoxMin || intersection;
                
				for (int whichSon = startSon; whichSon < endSon + 1; ++whichSon) {
					unsigned int indexToAdd = leftSonIndex(index) + whichSon;
                    
					if (indexToAdd < tPoints) {
                        			push_back(&indecesToVisit,indexToAdd);
					}
				}
			}
		}
		
		__syncthreads ();
		if (thread_index == 0)       
			results[point_index] = pointsFound[group_index];
	}    
}

void CUDAKernelWrapper(unsigned int nPoints,float *d_dim,unsigned int *d_ids,unsigned int *d_results)
{


    // Number of thread blocks
    unsigned int gridSize = (int)ceil((float)nPoints*NTREES/BLOCKSIZE);
    
    CUDASearchInTheKDBox<<<gridSize, BLOCKSIZE>>>(nPoints, d_dim, d_ids,d_results);


    
}

