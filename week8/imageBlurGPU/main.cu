/*
 * =====================================================================================
 *
 *       Filename:  main.c
 *
 *    Description:  Ch03 Samples
 *
 *        Version:  1.0
 *        Created:  07/14/2021 10:41:21 PM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Yoon, Myung Kuk, myungkuk.yoon@ewha.ac.kr
 *   Organization:  EWHA Womans Unversity
 *
 * =====================================================================================
 */

#include<iostream>
#include "ppm.h"
#include "clockMeasure.h"

#define checkCudaError(error) 					\
	if(error != cudaSuccess){ 				\
		printf("%s in %s at line %d\n", 		\
				cudaGetErrorString(error), 	\
				__FILE__ ,__LINE__); 		\
		exit(EXIT_FAILURE);				\
	}

#define BLUR_SIZE 5

using namespace std;

const int MAX_ITER = 1; //코드짤 때는 1로 설정, cpu와 gpu와의 성능 차이를 계산할 때는 10 -> 100으로 바꿔보면서 테스트


void cpuCode(unsigned char *outArray, const unsigned char *inArray, const int w, const int h){
	for(int row=0; row<h; row++){
		for(int col=0; col<w; col++){
			float avgR = 0.0f;
			float avgG = 0.0f;
			float avgB = 0.0f;
			
			int pixels = 0;
			int index = 0;

			for(int rowOffset = -BLUR_SIZE; rowOffset < BLUR_SIZE+1; rowOffset++){
				for(int colOffset = -BLUR_SIZE; colOffset < BLUR_SIZE+1; colOffset++){
					int curRow = row + rowOffset;
					int curCol = col + colOffset;
	
					if(curRow >= 0 && curRow < h && curCol >= 0 && curCol < w){
						int curIndex = (curRow * w + curCol) * 3;
						avgR += inArray[curIndex];
						avgG += inArray[curIndex+1];
						avgB += inArray[curIndex+2];
						pixels++;
					}
				}
			}

			avgR = (unsigned char)(avgR/pixels);
			avgG = (unsigned char)(avgG/pixels);
			avgB = (unsigned char)(avgB/pixels);
			
			index = (row * w + col) * 3;
			outArray[index] = avgR;
			outArray[index+1] = avgG;
			outArray[index+2] = avgB;
		}
	}
}


//Exercise: Write your GPU code.
__global__
void gpuCode(unsigned char *outArray, const unsigned char *inArray, const int w, const int h){
	//하나의 스레드 블락의 크기 = 16*16 = 265개의 스레드로 구성
	//그러한 스레드 블락의 개수 (w/265)*(h/265) = gridSize
	//총 스레드의 개수 gridSize * blockSize = (w/265)*(h/265)*(16*16)
	//스레의 개수만큼의 아이디가 있어야 하고 각각의 스레드를 각각의 픽셀로 할당해야 한다

	//int tld = blockIdx.x + blockIdx.y * gridDim.x; //스레드의 번호 및 픽셀의 번호
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	float avgR = 0.0f;
	float avgG = 0.0f;
	float avgB = 0.0f;		
	int pixels = 0;

	if (row < h && col < w){
		for (int rowOffset = -BLUR_SIZE; rowOffset < BLUR_SIZE + 1; rowOffset++){
			for (int colOffset = -BLUR_SIZE; colOffset < BLUR_SIZE + 1; colOffset++){
				int curRow = row + rowOffset;
				int curCol = col + colOffset;

				if(curRow >= 0 && curRow < h && curCol >= 0 && curCol < w){
					int curIndex = (curRow * w + curCol) * 3;
					avgR += inArray[curIndex];
					avgG += inArray[curIndex+1];
					avgB += inArray[curIndex+2];
					pixels++;
				}
			}
		}

		avgR = (unsigned char)(avgR/pixels);
		avgG = (unsigned char)(avgG/pixels);
		avgB = (unsigned char)(avgB/pixels);
		int index = (row * w + col) * 3;
		outArray[index] = avgR;
		outArray[index+1] = avgG;
		outArray[index+2] = avgB;
	}
}

int main(){
	int w, h;
	unsigned char *h_imageArray;
	unsigned char *h_outImageArray;
	unsigned char *d_imageArray;
	unsigned char *d_outImageArray;
	unsigned char *h_outImageArray2;

	ppmLoad("./data/ewha_picture.ppm", &h_imageArray, &w, &h);

	size_t arraySize = sizeof(unsigned char) * h * w * 3;

	h_outImageArray = (unsigned char*)malloc(arraySize);
	h_outImageArray2 = (unsigned char*)malloc(arraySize);

	cudaError_t err = cudaMalloc((void **) &d_imageArray, arraySize);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_outImageArray, arraySize);
	checkCudaError(err);

	err = cudaMemcpy(d_imageArray, h_imageArray, arraySize, cudaMemcpyHostToDevice);
	checkCudaError(err);

	const int tSize = 16;
	dim3 blockSize(tSize, tSize, 1);
	dim3 gridSize(ceil((float)w/tSize), ceil((float)h/tSize), 1);

	clockMeasure *ckCpu = new clockMeasure("CPU CODE");
	clockMeasure *ckGpu = new clockMeasure("GPU CODE");

	ckCpu->clockReset();
	ckGpu->clockReset();

	for(int i = 0; i < MAX_ITER; i++){
		
		ckCpu->clockResume();
		cpuCode(h_outImageArray, h_imageArray, w, h);
		ckCpu->clockPause();

		ckGpu->clockResume();
		gpuCode<<<gridSize, blockSize>>>(d_outImageArray, d_imageArray, w, h);
		err=cudaDeviceSynchronize();
		ckGpu->clockPause();
		checkCudaError(err);

	}
	ckCpu->clockPrint();
	ckGpu->clockPrint();

	err = cudaMemcpy(h_outImageArray2, d_outImageArray, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);
			
	ppmSave("ewha_picture_cpu.ppm", h_outImageArray, w, h);
	ppmSave("ewha_picture_gpu.ppm", h_outImageArray2, w, h);

	cudaFree(d_imageArray);
	cudaFree(d_outImageArray);
	
	return 0;
}
