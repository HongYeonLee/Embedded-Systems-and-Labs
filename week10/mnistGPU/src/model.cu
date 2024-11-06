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
		free((*iter)->result);

		cudaFree((*iter)->d_weights);
		cudaFree((*iter)->d_bias);
		cudaFree((*iter)->d_result);

		free((*iter));
	}
	m_layers.clear();
}

bool model::read_weights(const char *file, bool activation){
	bool ret = true;
	FILE *fp = fopen(file, "r");
	
	unsigned x = 0, y = 0;
	fLayer *layer = nullptr;

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

	//Copying Data from Host to Device
	this->copy_weights_into_device(layer);

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
	for(int w = 0; w < layer->num_weight[1]; w++){
		layer->result[w] = layer->bias[w].fp;
	}

	for(int i = 0; i < layer->num_weight[1]; i++){
		for(int w = 0; w < layer->num_weight[0]; w++){
			unsigned idx = i * layer->num_weight[0] + w;
			layer->result[i] += img[w] * layer->weights[idx].fp;
		}	
	}
}

void model::perf_act_exec(fLayer *layer){
	for(int w = 0; w < layer->num_weight[1]; w++){
		if(layer->result[w] < 0.0){
			layer->result[w] = 0.0f;
		}
	}
}


void model::copy_weights_into_device(fLayer *layer){
	layer->weightSize = sizeof(float) * layer->num_weight[0] * layer->num_weight[1];
	cudaError_t err = cudaMalloc((void **)&layer->d_weights, layer->weightSize);
	checkCudaError(err);
	layer->biasSize = sizeof(float) * layer->num_bias[0] * layer->num_bias[1];
	err = cudaMalloc((void **)&layer->d_bias, layer->biasSize);
	checkCudaError(err);

	layer->resultSize = sizeof(float) * layer->num_weight[1];
	err = cudaMalloc((void **)&layer->d_result, layer->resultSize);
	checkCudaError(err);

	err = cudaMemcpy(layer->d_weights, layer->weights, layer->weightSize, cudaMemcpyHostToDevice);
	checkCudaError(err);
	err = cudaMemcpy(layer->d_bias, layer->bias, layer->biasSize, cudaMemcpyHostToDevice);
	checkCudaError(err);
}

__global__
void perf_fc_exec_device(float *input, float *weight, float *bias, float *result, const unsigned weightSizeY, const unsigned weightSizeX, const unsigned biasSizeY, const unsigned biasSizeX){
	//Exercise: Write the code to perform a fully connected layer
	int tld = blockIdx.x * blockDim.x + threadIdx.x;

	if (tld < weightSizeY){ //아웃풋이 총 100개 존재하는데 스레드는 128개 존재하니 범위 체크 필요
		//바이어스 값으로 초기화
		result[tld] = bias[tld];

		for (int i = 0; i < weightSizeX; i++){ //784
			//바이어스 값에 접근하기 위한 인덱스 계산, 0번째 결과물에는 weight 배열의 0행의 784개의 weight값 접근 필요
			//n번째 결과물에는 weight배열의 n행의 784개의 weight값 접근 필요, 즉 n번째 행의 0번째 weight의 인덱스는 n * 784 + 0
			//tld가 col과 같은 역할
			int curIdx = tld * weightSizeX + i; 
			result[tld] += input[i] * weight[curIdx];
		}
	}
}

__global__
void perf_act_exec_device(float *input, unsigned inputSize){
	//Exercise: write the code to perform a activation function
	for (int i = 0; i < inputSize; i++){ //100
		if (input[i] < 0.0f){
			input[i] = 0.0f;
		}
	}
}

unsigned char model::perf_forward_exec_on_device(m_data *img){
	unsigned inputSize = sizeof(float) * IMG_SIZE * IMG_SIZE;
	float *d_input;
	cudaError_t err = cudaMalloc((void **)&d_input, inputSize);
	checkCudaError(err);

	err = cudaMemcpy(d_input, img->nor_data.oneD, inputSize, cudaMemcpyHostToDevice);
	checkCudaError(err);

	return perf_forward_exec_on_device(d_input);
}

unsigned char model::perf_forward_exec_on_device(float *d_input){
	float *input = d_input;
	for(auto iter = m_layers.begin(); iter != m_layers.end(); iter++){
		const unsigned tbSize = 128;
		dim3 blockSize(tbSize, 1, 1);
		dim3 gridSize(ceil((float)(*iter)->num_weight[1]/tbSize), 1, 1);
		perf_fc_exec_device<<<gridSize, blockSize>>>(input, (*iter)->d_weights, (*iter)->d_bias, (*iter)->d_result, (*iter)->num_weight[1], (*iter)->num_weight[0], (*iter)->num_bias[1], (*iter)->num_bias[0]);
		if((*iter)->perf_act){
			perf_act_exec_device<<<gridSize, blockSize>>>((*iter)->d_result, (*iter)->num_bias[0]);
		}
		cudaError_t err=cudaDeviceSynchronize();
		checkCudaError(err);
		input = (*iter)->d_result;
	}

	float h_input[10];
	//Exercise: copy the result from device memory to host memory
	//cudaError_t err = cudaMemcpy(...);
	//checkCudaError(err);
	cudaError_t err = cudaMemcpy(h_input, input, 10*sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaError(err);

	unsigned char largestIdx = 0;
	float largestO = h_input[0];
	for(unsigned char i = 1; i < 10; i++){
		if(h_input[i] > largestO){
			largestIdx = i;
			largestO = h_input[i];
		}
	}
	return largestIdx;
}
