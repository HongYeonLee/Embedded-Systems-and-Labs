/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description: 	Matrix Multiplication
 *
 *        Version:  1.0
 *        Created:  2021/07/30 10:07:38
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  이홍연
			   ID:	2371049
 *   Organization:  Ewha Womans University
 *
 * =====================================================================================
 */

#include <assert.h>
#include "clockMeasure.h"

#define checkCudaError(error) 					\
	if(error != cudaSuccess){ 				\
		printf("%s in %s at line %d\n", 		\
				cudaGetErrorString(error), 	\
				__FILE__ ,__LINE__); 		\
		exit(EXIT_FAILURE);				\
	}

const int A_H = 512;
const int A_W = 512;
const int B_H = A_W;
const int B_W = 512;
const unsigned int MAX_NUM = 100;
const int MAX_ITER = 1;
const int NUM_STREAMS = 4;

unsigned int matrixA[A_H * A_W];
unsigned int matrixB[B_H * B_W];
unsigned int cpuOut[A_H * B_W];
unsigned int gpuOut[A_H * B_W];
unsigned int streamGpuOut[A_H * B_W];

void generateRandomValues(unsigned int *input, const int rowSize, const int colSize){
	for(int i = 0; i < rowSize; i++){
		for(int j = 0; j < colSize; j++){
			input[i * colSize + j] = (unsigned int) float(rand())/float(RAND_MAX) * MAX_NUM;
		}
	}
}

void printMatrixValue(const unsigned int *input, const int rowSize, const int colSize){
	printf("Print Matrix \n -----------\n");
	for(int i = 0; i < rowSize; i++){
		for(int j = 0; j < colSize; j++){
			printf("%u\t", input[i * colSize + j]);
		}
		printf("\n");
	}
	printf("--------\n");
}

bool compareMatrix(const unsigned int *inputA, const unsigned int *inputB, const int rowSize, const int colSize){
	bool ret = true;
	for(int i = 0; i < rowSize * colSize; i++){
		if(inputA[i] != inputB[i]){
			ret = false;
			break;
		}
	}
	return ret;
}

void cpuMatrixMul(const unsigned int *h_a, const unsigned int *h_b, unsigned int *h_c, const int aRowSize, const int aColSize, const int bRowSize, const int bColSize){
	assert(aColSize == bRowSize);
	for(int i = 0; i < aRowSize; i++){
		for(int j = 0; j < bColSize; j++){
			unsigned int tSum = 0;
			for(int k = 0; k < aColSize; k++){
				tSum += (h_a[i * aColSize + k] * h_b[k * bColSize + j]);
			}
			h_c[i * bColSize + j] = tSum;
		}
	}
}

__global__
void gpuMatrixMul(unsigned int *d_a, unsigned int *d_b, unsigned int *d_c, const int aRowSize, const int aColSize, const int bRowSize, const int bColSize){
	assert(aColSize == bRowSize);
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(tId < aRowSize * bColSize){
		int rowId = tId / bColSize;
		int colId = tId % bColSize;
		unsigned int tSum = 0;
		for(int i = 0; i < aColSize; i++){
			tSum += (d_a[rowId * aColSize + i] * d_b[i * bColSize + colId]);
		}
		d_c[tId] = tSum;
	}
}

int main(){
	srand((unsigned int)time(NULL));
	generateRandomValues(matrixA, A_H, A_W);
	generateRandomValues(matrixB, B_H, B_W);

	unsigned int *d_a, *d_b, *d_c;
	size_t matrixSizeA = sizeof(unsigned int) * A_H * A_W;
	size_t matrixSizeB = sizeof(unsigned int) * B_H * B_W;
	size_t matrixSizeC = sizeof(unsigned int) * A_H * B_W;

	cudaError_t err = cudaMalloc((void **) &d_a, matrixSizeA);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_b, matrixSizeB);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_c, matrixSizeC);
	checkCudaError(err);

	err = cudaMemcpy(d_a, matrixA, matrixSizeA, cudaMemcpyHostToDevice);
	checkCudaError(err);
	err = cudaMemcpy(d_b, matrixB, matrixSizeB, cudaMemcpyHostToDevice);
	checkCudaError(err);

	const int tbSize = 256;
	dim3 gridSize(ceil((float)(A_H * B_W)/(float)tbSize), 1, 1);
	dim3 blockSize(tbSize, 1, 1);

	clockMeasure *ckCpu = new clockMeasure("CPU CODE");
	ckCpu->clockReset();
	
	clockMeasure *ckGpu = new clockMeasure("GPU CODE");
	ckGpu->clockReset();

	clockMeasure *ckGpuStream = new clockMeasure("GPU STREAM CODE");
	ckGpuStream->clockReset();

	for(int i = 0; i < MAX_ITER; i++){
		ckCpu->clockResume();
		cpuMatrixMul(matrixA, matrixB, cpuOut, A_H, A_W, B_H, B_W);
		ckCpu->clockPause();

		ckGpu->clockResume();
		gpuMatrixMul<<<gridSize, blockSize>>>(d_a, d_b, d_c, A_H, A_W, B_H, B_W);
		err=cudaDeviceSynchronize();
		ckGpu->clockPause();
		checkCudaError(err);
	}

	err = cudaMemcpy(gpuOut, d_c, matrixSizeC, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	cudaStream_t streams[NUM_STREAMS];
	for (int i = 0; i < NUM_STREAMS; i++){
		cudaStreamCreate(&streams[i]);
	}

	for (int j = 0; j < NUM_STREAMS; j++){
		unsigned idx = ((A_H/NUM_STREAMS) * j) * A_W;
		err = cudaMemcpyAsync(&d_a[idx], &matrixA[idx], matrixSizeA/NUM_STREAMS, cudaMemcpyHostToDevice, streams[j]);
		checkCudaError(err);
	}

	dim3 streamGridSize(((A_H/NUM_STREAMS) * B_W + blockSize.x - 1) / blockSize.x, 1, 1);
	dim3 streamBlockSize(tbSize, 1, 1);

	for (int i = 0; i < NUM_STREAMS; i++){
		unsigned idx = ((A_H/NUM_STREAMS) * i) * A_W;
		unsigned cIdx = ((A_H/NUM_STREAMS) * i) * B_W;
		ckGpuStream->clockResume();
		gpuMatrixMul<<<streamGridSize, streamBlockSize, 0, streams[i]>>>(&d_a[idx], d_b, &d_c[cIdx], A_H/NUM_STREAMS, A_W, B_H, B_W);
		ckGpuStream->clockPause();
	}

	for (int j = 0; j < NUM_STREAMS; j++){
		unsigned idx = ((A_H/NUM_STREAMS) * j) * B_W;
		err = cudaMemcpyAsync(&streamGpuOut[idx], &d_c[idx], matrixSizeC/NUM_STREAMS, cudaMemcpyDeviceToHost, streams[j]);
		checkCudaError(err);
	}

	for (int i = 0; i < NUM_STREAMS; i++){
		cudaStreamDestroy(streams[i]);
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	if(compareMatrix(cpuOut, gpuOut, A_H, B_W) && compareMatrix(gpuOut, streamGpuOut, A_H, B_W)){
		ckCpu->clockPrint();
		ckGpu->clockPrint();
		ckGpuStream->clockPrint();
	}else{
		printf("ERROR: Three Matrices are not same\n");
	}
}
