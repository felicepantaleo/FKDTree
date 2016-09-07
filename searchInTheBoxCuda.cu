/*! \file searchInTheBoxCuda.cu
*CUDA file with search methods
*/

#define MAX_SIZE 128
#define NUM_DIMENSIONS 3
#define MAX_RESULT_SIZE 128
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

/*! Size of merging buffer per thread group*/
#define MAX_FRONTIER 256
/*! Number of threads per thread group that are working in the same query*/
#define THREADS_PER_QUERY 16
/*! NUmber of queries per block. Derived by BLOCKSIZE and THREADS_PER_QUERY*/
#define QUERIES_PER_BLOCK (BLOCKSIZE/THREADS_PER_QUERY)

//! The k-d tree search function that uses work balancing
/*! This k-d tree search __global__ function performs box queries in the k-d tree. The k-d tree points are used for box centers. 
* Work balancing is done between the CUDA threads assigned to each query. THREADS_PER_QUERY, QUERIES_PER_BLOCK, BLOCKSIZE, MAX_FRONTIER 
* (maximum number of nodes in merged queue) defined in the file.
* Merge is done by finding write offsets with a Scan algorithm and then scattering the nodes there.
\param nPoints the number of points in the k-d tree
\param dimensions the coordinates of the points in the k-d tree
\param ids the identifiers of the points in the k-d tree
\param results the array with the results of the queries
\param share global memory assigned for the synchronization structures
*/
__global__ void CUDASearchInTheKDBox (unsigned int nPoints,  float* dimensions,  unsigned int* ids,  unsigned int* results, unsigned int* share) {
	unsigned int gid = threadIdx.x/THREADS_PER_QUERY; /*finds the local id of the query inside the block*/
	unsigned int pid = blockIdx.x*QUERIES_PER_BLOCK+gid; /*find the global id of the query*/
	unsigned int tid = threadIdx.x%THREADS_PER_QUERY; /*find the id of the thread in the in-query context*/

	unsigned int* frontier = share+nPoints*2*THREADS_PER_QUERY+pid*MAX_FRONTIER; /*search frontier-is used when merging the BFS queues*/
	unsigned int* frontnext =  share+nPoints*THREADS_PER_QUERY+pid*THREADS_PER_QUERY; /*holds offset for writing inside the frontier*/
	//unsigned int* pfound = share+pid*THREADS_PER_QUERY; /*number of results found per query*/

	__shared__ unsigned int pfound[QUERIES_PER_BLOCK];

	if (pid < nPoints) {
		unsigned int sync = 4;	/*sync at step 4*/
		/*get the box dimensions*/
		int theDepth = floor(log2((float)nPoints));
		float minPoint[NUM_DIMENSIONS];
		float maxPoint[NUM_DIMENSIONS];
		for(int i = 0; i<NUM_DIMENSIONS; ++i) {
			minPoint[i] = dimensions[nPoints*i+pid] - RANGE;
			maxPoint[i] = dimensions[nPoints*i+pid] + RANGE;
		}
		
		/*local queue*/
		Queue indecesToVisit;
		indecesToVisit.front = indecesToVisit.tail = indecesToVisit.size = 0;
		/*init thread 0*/
		if (tid == 0) {
			push_back(&indecesToVisit, 0);
			pfound[gid] = 0;
		}

		__syncthreads();

		unsigned int resultIndex = nPoints + MAX_RESULT_SIZE*pid;
	
		for (int depth = 0; depth < theDepth+1; ++depth) {
			if (depth == sync) {
				sync = sync + 3;

				/*initialize frontnext with number of elements in local queue*/
				frontnext[tid] = indecesToVisit.size;
				__syncthreads ();
				/*compute inclusive prefix scan*/
				for (unsigned int i = 1; i < THREADS_PER_QUERY; i<<=1) {
					int add;
        	               		if (tid >= i) add = frontnext[tid-i];
                        		__syncthreads();
                        		if (tid >= i) frontnext[tid] += add;
					__syncthreads();
	                	}
				/*share work in frontier*/
				unsigned int active = frontnext[THREADS_PER_QUERY-1];
	                        unsigned int len = indecesToVisit.size;
	                        unsigned int offset = frontnext[tid]-len;
			
				for (int i = 0; i < len; i++) {
					frontier[i+offset] = pop_front(&indecesToVisit);
				}
				__syncthreads ();

				/*get new work*/
				for (int i = tid; i < active; i+=THREADS_PER_QUERY) {
					push_back(&indecesToVisit, frontier[i]);
				}

			}		
			/*check next level of nodes set to be visited*/
			int dimension = depth % NUM_DIMENSIONS;
			unsigned int toVisit = indecesToVisit.size;
			
			for (unsigned int visited = 0; visited < toVisit; visited++) {
				unsigned int index = pop_front(&indecesToVisit);

				bool intersection = intersects(index, dimensions, nPoints, minPoint, maxPoint, dimension);

				if (intersection && isInTheBox(index,dimensions,nPoints,minPoint,maxPoint)) {
					if (pfound[gid] < MAX_RESULT_SIZE) {
						unsigned int found = atomicAdd(&pfound[gid],1);
						if (found < MAX_RESULT_SIZE)
							results[resultIndex+found] = index;
					}
				}
			
				bool isLowerThanBoxMin = dimensions[nPoints*dimension+index] < minPoint[dimension];
				int startSon = isLowerThanBoxMin;
				int endSon = startSon || intersection;

				for (int whichSon = startSon; whichSon < endSon + 1; ++whichSon) {
					unsigned int indexToAdd = leftSonIndex(index)+whichSon;
					if (indexToAdd < nPoints)
						push_back(&indecesToVisit, indexToAdd);
				}
			}			
		}
	
		__syncthreads();

		/*set as number of results*/
		if (tid == 0) {
			results[pid] = pfound[gid];
		}

		__syncthreads ();
	}
}

void CUDAKernelWrapper(unsigned int nPoints,float *d_dim,unsigned int *d_ids,unsigned int *d_results, unsigned int* debug)
{


    	/*thread block sizes*/
	unsigned int gridSize = (int)ceil((float)nPoints/QUERIES_PER_BLOCK);
    
	unsigned int blockSize = MAX_FRONTIER;

	CUDASearchInTheKDBox<<<gridSize, blockSize>>>(nPoints, d_dim, d_ids,d_results, debug);

	cudaError_t err = cudaGetLastError ();
	printf ("Error:%s\n", cudaGetErrorString (err));
    
}

