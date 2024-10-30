/*
 * =====================================================================================
 *
 *       Filename:  model.c
 *
 *    Description: To load weight from files and execute inference
 *
 *        Version:  1.0
 *        Created:  07/29/2024 12:46:39 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Myung Kuk Yoon (MK), myungkuk.yoon@ewha.ac.kr
 *   Organization:  Department of Computer Science and Engineering
 *
 * =====================================================================================
 */

#include "model.h"

model::model(unsigned num_layers){
	m_num_layers = num_layers;
}

model::~model(){
	for(auto iter = m_layers.begin(); iter != m_layers.end(); iter++){
		free((*iter)->weights);
		free((*iter)->bias);
		free((*iter));
	}
	m_layers.clear();
}

bool model::read_weights(const char *file, bool activation){
	bool ret = true;
	FILE *fp = fopen(file, "r");
	
	unsigned x = 0, y = 0;
	fLayer *layer = nullptr;
	//u2f tmp;


	if(!fp){
		printf("[ERROR] Weight Input file does not exist\n");
		ret = false;
		goto cleanup;
	}

	fscanf(fp, "%u x %u\n", &y, &x);

	if(!x && !y){
		printf("[ERROR] On weight dimension\n");
		ret = false;
		goto cleanup;
	}

	layer = (fLayer *)malloc(sizeof(fLayer));
	layer->num_weight[0] = y;
	layer->num_weight[1] = x;
	layer->weights = (u2f *)malloc(sizeof(u2f) * x * y);

	for(int i = 0; i < y; i++){
		for(int j = 0; j < x; j++){
			fscanf(fp, "0x%x ", &layer->weights[j * y + i].uint);
		}
	}

	fscanf(fp, "%u x %u\n", &y, &x);
	if((!x && !y) || x != 1){ 
		printf("[ERROR] On bias dimension\n");
		free(layer->weights);
		free(layer);
	}

	layer->num_bias[0] = y;
	layer->num_bias[1] = x;
	layer->bias = (u2f *)malloc(sizeof(u2f) * y * x);

	for(int i = 0; i < y; i++){
		fscanf(fp, "0x%x ", &layer->bias[i].uint);
	}
	
	assert(layer->num_weight[1] == layer->num_bias[0]);
	layer->result = (float *)malloc(sizeof(float) * layer->num_weight[1]);
	
	layer->perf_act = activation;

	m_layers.push_back(layer);
	printf("Loaded Layer %u = W:(%u x %u) + B:(%u x %u)\n", (unsigned)m_layers.size(), layer->num_weight[0], layer->num_weight[1], layer->num_bias[0], layer->num_bias[1]);

cleanup:
	fclose(fp);
	
	return ret;
}

unsigned char model::perf_forward_exec(m_data *img){
	float *input = img->nor_data.oneD;
	return perf_forward_exec(input);
}

unsigned char model::perf_forward_exec(float *input){
	for(auto iter = m_layers.begin(); iter != m_layers.end(); iter++){
		perf_fc_exec((*iter), input);	
		if((*iter)->perf_act) perf_act_exec((*iter));
		input = (*iter)->result;
	}
	unsigned char largestIdx = 0;
	float largestO = input[0];
	for(unsigned char i = 1; i < 10; i++){
		if(input[i] > largestO){
			largestIdx = i;
			largestO = input[i];
		}
	}
	return largestIdx;
}

void model::perf_fc_exec(fLayer *layer, float *img){
	//Exercise: Write code for fully-connectioned layer
	//You must store the computed values in the result variable within the fLayer structure.

	//result값 초기화
	for (int i = 0; i < layer->num_weight[1]; i++){
		layer->result[i] = 0;
	}

	//가중치 곱과 덧셈
	for (int i = 0; i < layer->num_weight[1]; i++){ //100
		for (int j = 0; j < layer->num_weight[0]; j++){ //784
			layer->result[i] += layer->weights[i * layer->num_weight[0] + j].fp * img[j]; //인덱스 = 행 * 784 + 열
		}
	}

	//바이어스값 덧셈
	for (int i = 0; i < layer->num_bias[0]; i++){
		layer->result[i] += layer->bias[i].fp;
	}
}

void model::perf_act_exec(fLayer *layer){
	//Exercise: Write code for activation layer
	//You must store the computed values in the result variable within the fLayer structure.
	for (int i = 0; i < layer->num_weight[1]; i++) { //100
		if (layer->result[i] < 0){
			layer->result[i] = 0;
		}
	}
}
