/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description:  Fully-Connected MNIST
 *
 *        Version:  1.0
 *        Created:  07/08/2024 09:54:25 AM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  이홍연
 * 			   ID:	2371049
 *   Organization:  EWHA Womans University
 *
 * =====================================================================================
 */


#include "./src/predefine.h"
#include "./src/dataReader.h"
#include "./src/model.h"
#include "./src/clockMeasure.h"
#define imgSize 784
#define batchSize 1000

const char *train_image = "./data/train-images";
const char *train_label = "./data/train-labels";
const char *test_image = "./data/test-images";
const char *test_label = "./data/test-labels";

int main(){
	//Reading train and test images
	dataReader *reader = new dataReader(train_image, train_label, test_image, test_label);

	reader->print_file_info();

	reader->read_train_files();
	reader->read_test_files();

	reader->calculate_std_mean();
	reader->apply_nor_into_trainDB();
	reader->apply_nor_into_testDB();

	//Print image and label
	reader->print_image_and_label(12, false); 

	//Create FC NN (Read Weight)
	model *fModel = new model(2);
	fModel->read_weights("./layer/Layer01.db");
	fModel->read_weights("./layer/Layer02.db", false);

	clockMeasure *ckCpu = new clockMeasure("CPU CODE");
	ckCpu->clockReset();
	clockMeasure *ckGpu = new clockMeasure("GPU CODE");
	ckGpu->clockReset();

	//Get image info and perform inference
	for(unsigned i = 0; i < 1; i++){
		m_data *img[imgSize * batchSize]; //1차원 배열, 1000개의 이미지 데이터를 가지고 있음
		for (int j = 0; j < batchSize; j++){
			int imgIdx = j * imgSize;
			img[imgIdx] = reader->get_mnist_db(j, false);
		}
		
		ckCpu->clockResume();
		unsigned cpu_ret = fModel->perf_forward_exec(img);
		ckCpu->clockPause();

		ckGpu->clockResume();
		unsigned gpu_ret = fModel->perf_forward_exec_on_device(img);
		ckGpu->clockPause();
		if(cpu_ret != gpu_ret){
			printf("Error: results do not match for index %u\n", i);
		}
	}

	ckCpu->clockPrint();
	ckGpu->clockPrint();

	return 0;
}
