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
	for (int row = 0; row < h; row++){
		for (int col  =0; col < w; col++){
			float sumR = 0.0f;
			float sumG = 0.0f;
			float sumB = 0.0f;
			int pixels = 0;
			
			for (int rowOffest = -BLUR_SIZE; rowOffest < BLUR_SIZE + 1; rowOffest++){
				for (int colOffset = -BLUR_SIZE; colOffset < BLUR_SIZE + 1; colOffset++){
					int curRow = row + rowOffest; //작업중인 픽셀을 기준으로 -5 ~ +5 한 픽셀
					int curCol = col + colOffset; //작업중인 픽셀을 기준으로 -5 ~ +5 한 픽셀

					if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w){
						int curIndex = (curRow*w + curCol)*3; //작업중인 픽셀을 기준으로 -5 ~ +5 한 픽셀의 RGB값에 접근하기 위한 인덱스
						sumR += inArray[curIndex];
						sumG += inArray[curIndex + 1];			
						sumB += inArray[curIndex + 2];
						pixels++;	
					}
				}
			}

			unsigned char avgR = (unsigned char)(sumR / pixels);
			unsigned char avgG = (unsigned char)(sumG / pixels);
			unsigned char avgB = (unsigned char)(sumB / pixels);

			int index = (row*w + col)*3; //현재 픽셀의 번호에서, 그 픽셀의 RGB값에 접근하기 위해서 *3을 함
			outArray[index] = avgR;
			outArray[index + 1] = avgG;
			outArray[index + 2] = avgB;
		}
	}
}

int main(){
	int w, h;
	unsigned char *h_imageArray; //포인터로 선언
	unsigned char *h_outImageArray; //포인터로 선언

	//This function will load the R/G/B values from a PPM file into an array and return the width (w) and height (h).
	ppmLoad("./data/ewha_picture.ppm", &h_imageArray, &w, &h);
	h_outImageArray = (unsigned char*)malloc(w*h*3*sizeof(unsigned char));

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

	free(h_imageArray);
	free(h_outImageArray);

	return 0;
}
