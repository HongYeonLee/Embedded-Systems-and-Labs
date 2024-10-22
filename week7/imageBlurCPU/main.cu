/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description:  image blur cpu code
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

#define BLUR_SIZE 5

using namespace std;

const int MAX_ITER = 10;

void cpuCode(unsigned char *outArray, const unsigned char *inArray, const int w, const int h){
	//Exercise: Write image blur
	for (int i = 0; i < w*h*3; i++){
		int Rsum = 0, Gsum = 0, Bsum = 0;
		int index = i;
		//자기 자신값 더하기
		Rsum += inArray[index];
		Gsum += inArray[index + 1];
		Bsum += inArray[index + 2];

		//오른쪽으로 5칸 더하기
		for (int j = 0; j < BLUR_SIZE; j++){
			index += 3; //오른쪽으로 한칸 이동
			if(index < 0 || index > w*h*-1){

			}
			Rsum += inArray[index];
			Gsum += inArray[index + 1];
			Bsum += inArray[index + 2];
		}
		//왼쪽으로 5칸 더하기
		for (int j = 0; j < BLUR_SIZE; j++){
			index -= 3; //왼쪽으로 한칸 이동
			Rsum += inArray[index];
			Gsum += inArray[index + 1];
			Bsum += inArray[index + 2];
		}
		//위쪽으로 5칸 더하기
		for (int j = 0; j < BLUR_SIZE; j++){
			index -= 9; //위쪽으로 한칸 이동
			Rsum += inArray[index];
			Gsum += inArray[index + 1];
			Bsum += inArray[index + 2];
		}
		//아래쪽으로 5칸 더하기
		for (int j = 0; j < BLUR_SIZE; j++){
			index += 9; //아래쪽으로 한칸 이동
			Rsum += inArray[index];
			Gsum += inArray[index + 1];
			Bsum += inArray[index + 2];
		}

		//오른쪽 대락선 아래쪽으로 5칸 더하기
		for (int j = 0; j < BLUR_SIZE; j++){
			index += 12; //오른쪽 대각선 아래로 한칸 이동
			Rsum += inArray[index];
			Gsum += inArray[index + 1];
			Bsum += inArray[index + 2];
		}

		//오른쪽 대락선 위쪽으로 5칸 더하기
		for (int j = 0; j < BLUR_SIZE; j++){
			index -= 12; //오른쪽 대각선 위로 한칸 이동
			Rsum += inArray[index];
			Gsum += inArray[index + 1];
			Bsum += inArray[index + 2];
		}

		//왼쪽 대락선 위쪽으로 5칸 더하기
		for (int j = 0; j < BLUR_SIZE; j++){
			index -= 6; //왼쪽 대각선 위로 한칸 이동
			Rsum += inArray[index];
			Gsum += inArray[index + 1];
			Bsum += inArray[index + 2];
		}

		//왼쪽 대락선 아래쪽으로 5칸 더하기
		for (int j = 0; j < BLUR_SIZE; j++){
			index += 6; //왼쪽 대각선 아래로 한칸 이동
			Rsum += inArray[index];
			Gsum += inArray[index + 1];
			Bsum += inArray[index + 2];
		}

		inArray[i] = Rsum/(11*11);
		inArray[i+1] = Gsum/(11*11);
		inArray[i+2] = Bsum/(11*11); 

	}
}

int main(){
	int w, h;
	unsigned char *h_imageArray;
	unsigned char *h_outImageArray;

	//This function will load the R/G/B values from a PPM file into an array and return the width (w) and height (h).
	ppmLoad("./data/ewha_picture.ppm", &h_imageArray, &w, &h);
	h_outImageArray = ()
	cpuCode(h_outImageArray, h_imageArray, w, h);

	clockMeasure *ckCpu = new clockMeasure("CPU CODE");

	ckCpu->clockReset();

	for(int i = 0; i < MAX_ITER; i++){
		ckCpu->clockResume();
		cpuCode(h_outImageArray, h_imageArray, w, h); 
		ckCpu->clockPause();
	}
	ckCpu->clockPrint();

	//This function will store the R/G/B values from h_outImageArray into a PPM file.
	ppmSave("ewha_picture_cpu.ppm", h_outImageArray, w, h);

	return 0;
}
