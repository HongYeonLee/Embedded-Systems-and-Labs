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

	unsigned int *d_a, *d_b, *d_c, *d_d;
	size_t matrixSizeA = sizeof(unsigned int) * A_H * A_W;
	size_t matrixSizeB = sizeof(unsigned int) * B_H * B_W;
	size_t matrixSizeC = sizeof(unsigned int) * A_H * B_W;

	cudaError_t err = cudaMallocManaged((void **) &d_a, matrixSizeA);
	checkCudaError(err);
	err = cudaMallocManaged((void **) &d_b, matrixSizeB);
	checkCudaError(err);
	err = cudaMallocManaged((void **) &d_c, matrixSizeC); //cpu 결과
	checkCudaError(err);
	err = cudaMallocManaged((void **) &d_d, matrixSizeC); //gpu 결과
	checkCudaError(err);

	generateRandomValues(d_a, A_H, A_W);
	generateRandomValues(d_b, B_H, B_W);

	const int tbSize = 256;
	dim3 gridSize(ceil((float)(A_H * B_W)/(float)tbSize), 1, 1);
	dim3 blockSize(tbSize, 1, 1);

	clockMeasure *ckCpu = new clockMeasure("CPU CODE");
	ckCpu->clockReset();
	
	clockMeasure *ckGpu = new clockMeasure("GPU CODE");
	ckGpu->clockReset();

	for(int i = 0; i < MAX_ITER; i++){
		ckCpu->clockResume();
		cpuMatrixMul(d_a, d_b, d_c, A_H, A_W, B_H, B_W);
		ckCpu->clockPause();

		ckGpu->clockResume();
		gpuMatrixMul<<<gridSize, blockSize>>>(d_a, d_b, d_d, A_H, A_W, B_H, B_W);
		err=cudaDeviceSynchronize();
		ckGpu->clockPause();
		checkCudaError(err);
	}

	if(compareMatrix(d_c, d_d, A_H, B_W)){
		ckCpu->clockPrint();
		ckGpu->clockPrint();
	}else{
		printf("ERROR: Two Matrices are not same\n");
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_d);
}
