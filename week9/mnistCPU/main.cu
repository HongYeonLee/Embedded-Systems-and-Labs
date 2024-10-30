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
 *         Author:  Myung Kuk Yoon 
 *   Organization:  EWHA Womans University
 *
 * =====================================================================================
 */


#include "./src/predefine.h"
#include "./src/dataReader.h"
#include "./src/model.h"
#include "./src/clockMeasure.h"

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



	//Create FC NN (Read Weight)
	model *fModel = new model(2);
	fModel->read_weights("./layer/Layer01.db");
	fModel->read_weights("./layer/Layer02.db", false);

	for (int i = 0; i < 100; i ++){
		//Print image and label
		reader->print_image_and_label(i, false); 
		//Get image info and perform inference
		m_data *img = reader->get_mnist_db(i, false);
		unsigned ret = fModel->perf_forward_exec(img);
		printf("Predicted Value: %u\n", ret);
	}

	return 0;
}
