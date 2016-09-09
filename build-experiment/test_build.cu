#include <chrono>
#include <sstream>
#include <unistd.h>
#include <thread>
#include "tbb/tbb.h"
#include <atomic>
#include <string.h>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"

#include <stdlib.h>
#include <sys/time.h>


/*USES TBB::TICK_COUNT FOR MEASURING TIME, use -t argument*/

#define FLOOR_LOG2(X) ((unsigned int)(log2((float)X)))

__device__ unsigned int partition_complete_kdtree (unsigned int len) {
	if (len == 1)
		return 0;

	unsigned int index = 1 << (FLOOR_LOG2(len));
	if ((index/2)-1 <= len-index)
		return index-1;
	else
		return len-index/2;
}

/*partition for smaller problems, uses quickselect*/
__global__ void nth_element_small (unsigned int nPoints, unsigned int dim, unsigned int depth,
			float* src_dim, float* dest_dim, unsigned int* src_id, unsigned int* dest_id,
			unsigned int* groupStart, unsigned int* groupLen) {
	uint tidx = blockIdx.x*blockDim.x+threadIdx.x;

	uint ls = (1 << depth) -1;

	uint le = 2*ls;

	for (int i = ls+tidx; i <= le && i < nPoints; i += gridDim.x*blockDim.x) {
		uint len = groupLen[i];
		
		if (len == 0) {
			continue;
		} else if (len == 1) {
			dest_id[i] = src_id[groupStart[i]];
			for (int d = 0; d < 3; d++)
				dest_dim[d*nPoints+i] = src_dim[d*nPoints+groupStart[i]];
		} else {
			uint offset = groupStart[i];
			uint start = groupStart[i];
			uint end = start+groupLen[i];
			uint target = partition_complete_kdtree (groupLen[i]);

			while (start < end) {
				uint pivot = start;
				uint idx = start+1;
				for (int j = start+1; j < end; j++) {
					if (src_dim[dim*nPoints+pivot] >= src_dim[dim*nPoints+j]) {
						for (int d = 0; d < 3; d++) {					
							float tmpf = src_dim[d*nPoints+j];
							src_dim[d*nPoints+j] = src_dim[d*nPoints+idx];
							src_dim[d*nPoints+idx] = tmpf;
						}
						int tmpi = src_id[j];
						src_id[j] = src_id[idx];
						src_id[idx] = tmpi;

						idx++;
					}
				}

				idx--;

				for (int d = 0; d < 3; d++) {
					float tmpf = src_dim[d*nPoints+pivot];
					src_dim[d*nPoints+pivot] = src_dim[d*nPoints+idx];
					src_dim[d*nPoints+idx] = tmpf;
				}
				int tmpi = src_id[pivot];
				src_id[pivot] = src_id[idx];
				src_id[idx] = tmpi;

				if (idx > target+offset)
					end = idx;
				else if (idx < target+offset)
					start = idx+1;
				else break;
			}

			if (2*i+1 < nPoints) {
				groupStart[2*i+1] = groupStart[i];
				groupLen[2*i+1] = target;
			}
			if (2*i+2 < nPoints) {
				groupStart[2*i+2] = groupStart[i]+target+1;
				groupLen[2*i+2] = groupLen[i]-target-1;
			}

			dest_id[i] = src_id[target+offset];
			for (int d = 0; d < 3; d++)
				dest_dim[d*nPoints+i] = src_dim[d*nPoints+target+offset];
		}
	}
}

/*partition for larger problems, is variation of quickselect*/
__global__ void nth_element (unsigned int nPoints, unsigned int dim, unsigned int depth,
			float* dimensions, unsigned int* ids, float* src_dim, float* tmp_dim, float* tmp_dim2,
			float* dest_dim, unsigned int* src_id,  unsigned int* dest_id, 
			unsigned int* groupStart, unsigned int* groupLen, unsigned int* global_hist) {

	#define BUCKETCOUNT 16

	__shared__ int selectedBucket;

	uint const tidx = threadIdx.x;
	uint const bsize = blockDim.x;
	uint const gnum = gridDim.x;

	uint ls = (1 << depth) -1;
	uint le = 2*ls;

	for (uint gid = ls+blockIdx.x; gid <= le && gid < nPoints; gid += gnum) {
		uint start = groupStart[gid];
		uint len = groupLen[gid];
		uint target = partition_complete_kdtree(len);

		float* dsrc = src_dim+start+dim*nPoints;
		float* ddst = tmp_dim+start;
		
		if (len == 0)
			continue;
		
		unsigned int* hist = global_hist+blockIdx.x*bsize*BUCKETCOUNT;

		uint local_hist[BUCKETCOUNT];
		
		for (uint bits = 0; bits < sizeof(uint)*2; bits++) {
			if (len == 1)
				break;

			uint lowBit = sizeof(float) * 8 - 4 - bits * 4;
			uint highBit = sizeof(float) * 8 - bits * 4;
			unsigned int diff = highBit-lowBit;
			uint mask = (1 << (diff)) - 1;

			for (uint i = 0; i < BUCKETCOUNT; i++) {
				local_hist[i] = 0;
				if (tidx == 0)
					hist[i] = 0;
			}

			for (uint row = tidx; row < len; row += bsize) {
				uint key = (((uint) (RAND_MAX*(dsrc[row]/10.1))) >> lowBit) & mask;
				(local_hist[key])++;
			}

			for (uint i = 0; i < BUCKETCOUNT; i++)
				hist[tidx + i * bsize] = local_hist[i];

			__syncthreads();

			if (tidx == 0) {
				int sb_local = -1;
				unsigned int sum = 0;
				for (uint i = 0; i < BUCKETCOUNT * bsize; i++) {
					unsigned int val = hist[i];
					hist[i] = sum;
					if (sum > target && sb_local < 0) {
                                                sb_local = (i-1) / bsize;
                                        }
					sum += val;
				}
				if (sb_local == -1) 
					selectedBucket = 15;
				else
					selectedBucket = sb_local;
			}

			__syncthreads();

			for (uint i = 0; i < BUCKETCOUNT; i++)
				local_hist[i] = hist[tidx + i * bsize];

			for (uint row = tidx; row < len; row += bsize) {
				uint key = (((uint) (RAND_MAX*(dsrc[row]/10.1))) >> lowBit) & mask;
				if (key == selectedBucket) {
					uint dstIndex = local_hist[key] - hist[selectedBucket * bsize];
					ddst[dstIndex] = dsrc[row];
					(local_hist[key])++;
				}
			}

			__syncthreads();

			float* temp = dsrc;
			dsrc = ddst;
			ddst = temp;

			if (ddst == src_dim+start+dim*nPoints)
				ddst = tmp_dim2+start;

			if (selectedBucket == 15)
				len -= hist[selectedBucket * bsize];
			else
				len = hist[(selectedBucket + 1) * bsize] - hist[selectedBucket * bsize];

			target -= hist[selectedBucket * bsize];
		}

		float nth_element_value = dsrc[target];
		unsigned int buckets[3];
		buckets[0] = 0;
		buckets[1] = 0;
		buckets[2] = 0;

		
		for (uint row = tidx; row < groupLen[gid]; row += bsize) {
			uint key = (src_dim[row+start+dim*nPoints] > nth_element_value) ? 2 : 1;
			key = (src_dim[row+start+dim*nPoints] < nth_element_value) ? 0 : key;
			buckets[key]++;
		}

		hist[tidx] = buckets[0];
		hist[tidx + bsize] = buckets[1];
		hist[tidx + 2 * bsize] = buckets[2];

		__syncthreads ();

		if (tidx == 0) {
			uint sum = 0;
			for (uint i = 0; i < 3 * bsize; i++) {
				uint t = hist[i];
				hist[i] = sum;
				sum += t;
			}
		}

		__syncthreads ();
		
		buckets[0] = hist[tidx];
		buckets[1] = hist[tidx + bsize];
		buckets[2] = hist[tidx + 2 * bsize];

		for (uint row = tidx; row < groupLen[gid]; row += bsize) {
			uint key = (src_dim[groupStart[gid]+dim*nPoints+row] > nth_element_value) ? 2 : 1;
			key = (src_dim[groupStart[gid]+dim*nPoints+row] < nth_element_value) ? 0 : key;
			for (uint d = 0; d < 3; d++) 
				dest_dim[groupStart[gid]+d*nPoints+buckets[key]] = src_dim[groupStart[gid]+d*nPoints+row];
			dest_id[groupStart[gid]+buckets[key]] = src_id[groupStart[gid]+row];
			(buckets[key])++;
		}

		__syncthreads ();

		if (tidx == 0) {
			uint leftChildIndex =  2*gid+1;
			uint rightChildIndex = 2*gid+2;
			target = partition_complete_kdtree(groupLen[gid]);

			if (leftChildIndex < nPoints) {
				groupStart[leftChildIndex] = groupStart[gid];
				groupLen[leftChildIndex] = target;
			}

			if (rightChildIndex < nPoints) {
				groupStart[rightChildIndex] = groupStart[gid] + target + 1;
				groupLen[rightChildIndex] = groupLen[gid] - target - 1;
			}

			ids[gid] = dest_id[groupStart[gid]+target];
			for (uint d = 0; d < 3; d++)
				dimensions[gid+d*nPoints] = dest_dim[groupStart[gid]+d*nPoints+target];
		}
		__syncthreads ();
	}
}


int test_kdtree (unsigned int* ids, float* dimensions, float min_point[3], float max_point[3], unsigned int depth, unsigned int index, unsigned int nPoints) {
	if (index >= nPoints)
		return 1;

	float min_point1[3]; float max_point1[3];
	float min_point2[3]; float max_point2[3];

	for (int d = 0; d < 3; d++) {
		min_point1[d] = min_point2[d] = min_point[d];
		max_point1[d] = max_point2[d] = max_point[d];

		if (dimensions[d*nPoints+index] > max_point[d] || dimensions[d*nPoints+index] < min_point[d]) {
			printf ("%f < %f < %f@d?\n", min_point[d], dimensions[d*nPoints+index], max_point[d], index);
			return -1;
		}
	}

	unsigned int dim = depth%3;
	max_point1[dim] = dimensions[dim*nPoints+index];
	min_point2[dim] = dimensions[dim*nPoints+index];

	if (test_kdtree (ids, dimensions, min_point1, max_point1, depth+1, 2*index+1, nPoints) < 0 ||
			test_kdtree (ids, dimensions, min_point2, max_point2, depth+1, 2*index+2, nPoints) < 0)
		return -1;

	return 1;
}

int main (int argc, char** argv) {
	if (argc != 2 && argc != 3) {
		printf ("Execute as ./exec numberofpoints [optional:-t for time]\n");
		exit(1);
	}

	int timer = 0;
	int nPoints = atoi(argv[1]);
	int nDimensions = 3;

	if (argc == 3) {
		if (strcmp(argv[2],"-t") == 0) {
			printf ("Time measuring enabled\n");
			timer = 1;
		} else
			exit(2);
	}

	srand(time(NULL));
	
	unsigned int* ids = (unsigned int*) malloc(nPoints*sizeof(int));
	float* dimensions = (float*) malloc(nDimensions*nPoints*sizeof(float));

	float minpoint[3];
	float maxpoint[3];

	for (int i = 0; i < nPoints; ++i)
	{
		ids[i] = i;

		float x = static_cast<float>(rand())
				/ (static_cast<float>(RAND_MAX / 10.1));

		float y = static_cast<float>(rand())
				/ (static_cast<float>(RAND_MAX / 10.1));

		float z = static_cast<float>(rand())
				/ (static_cast<float>(RAND_MAX / 10.1));

		dimensions[0*nPoints+i] = x;
		dimensions[1*nPoints+i] = y;
		dimensions[2*nPoints+i] = z;

		for (int j = 0; j < 3; j++) {
			if (minpoint[j] > dimensions[j*nPoints+i])
				minpoint[j] = dimensions[j*nPoints+i];
			if (maxpoint[j] < dimensions[j*nPoints+i])
				maxpoint[j] = dimensions[j*nPoints+i];
		}
	}

	/*variable declaration*/
	unsigned int* d_groupstart;
	unsigned int* d_grouplen;
	float* d_A;
	float* d_B;
	unsigned int* d_temp;
	float* d_points_src;
	float* d_points_src2;
	float* d_points_dst;
	unsigned int* d_ids_src;
	unsigned int* d_ids_src2;
	unsigned int* d_ids_dst;

	
	/*allocate memory for building*/
	cudaMalloc (&d_groupstart, nPoints*sizeof(unsigned int));
	cudaMalloc (&d_grouplen, nPoints*sizeof(unsigned int));
	cudaMalloc (&d_points_src, nDimensions*nPoints*sizeof(float));
	cudaMalloc (&d_points_src2, nDimensions*nPoints*sizeof(float));
	cudaMalloc (&d_points_dst, nDimensions*nPoints*sizeof(float));
	cudaMalloc (&d_ids_src, nPoints*sizeof(unsigned int));
	cudaMalloc (&d_ids_src2, nPoints*sizeof(unsigned int));
	cudaMalloc (&d_ids_dst, nPoints*sizeof(unsigned int));
	cudaMalloc (&d_A, nPoints*sizeof(float));
	cudaMalloc (&d_B, nPoints*sizeof(float));
	cudaMalloc (&d_temp, 64*64*16*sizeof (unsigned int));
	/*set starting values for groups*/
	{
		unsigned int x = 0;
		cudaMemcpy (d_groupstart, &x, sizeof(unsigned int), cudaMemcpyHostToDevice);
		x = nPoints;
		cudaMemcpy (d_grouplen, &x, sizeof(unsigned int), cudaMemcpyHostToDevice);
	}
	
	for (unsigned int d = 0; d < nDimensions; d++)
			cudaMemcpy (&d_points_src[d*nPoints], &dimensions[d*nPoints], nPoints*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy (d_ids_src, ids, nPoints*sizeof(unsigned int), cudaMemcpyHostToDevice);

	tbb::tick_count start_searching_CUDA = tbb::tick_count::now();

	unsigned int maximum_depth = ((unsigned int)(32 - __builtin_clz(nPoints | 1)));
	for (unsigned int depth = 0; depth < maximum_depth; depth++) {
		if (depth < 12) {
			nth_element <<<64,64>>> (nPoints, depth%nDimensions, depth, d_points_dst, d_ids_dst, 
					d_points_src, d_A, d_B, d_points_src2, d_ids_src,  d_ids_src2, d_groupstart, d_grouplen, d_temp);
			std::swap(d_points_src, d_points_src2);
			std::swap(d_ids_src, d_ids_src2);
		} else {
			nth_element_small <<<64,64>>> (nPoints, depth%nDimensions, depth, d_points_src, d_points_dst,
							d_ids_src, d_ids_dst, d_groupstart, d_grouplen); 
		}
		cudaStreamSynchronize(0);
	}

	cudaFree (d_A); cudaFree(d_B);
	cudaFree (d_groupstart); cudaFree (d_grouplen);


	cudaMemcpy (ids, d_ids_dst, nPoints*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	for (int d = 0; d < nDimensions; d++)
		cudaMemcpy (&dimensions[d*nPoints], &d_points_dst[d*nPoints], nPoints*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree (d_points_src); cudaFree(d_points_dst);cudaFree (d_ids_src); cudaFree(d_ids_dst);
		
	tbb::tick_count::interval_t end_searching_CUDA = tbb::tick_count::now()-start_searching_CUDA;
	
	if (timer)
		std::cout << "Building k-d tree using CUDA took "
			<< end_searching_CUDA.seconds()*1e3<< "ms\n"<<std::endl;	

	if (test_kdtree (ids, dimensions, minpoint, maxpoint, 0, 0, nPoints) < 0)
		printf ("Error!!!\n");

	return 0;
}
