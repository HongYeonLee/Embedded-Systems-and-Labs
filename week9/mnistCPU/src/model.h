/*
 * =====================================================================================
 *
 *       Filename:  model.h
 *
 *    Description:  To load weight from files and execute inference
 *
 *        Version:  1.0
 *        Created:  07/29/2024 12:41:09 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Myung Kuk Yoon (MK), myungkuk.yoon@ewha.ac.kr
 *   Organization:  Department of Computer Science and Engineering
 *
 * =====================================================================================
 */

#pragma once

#include "predefine.h"

class model{
	public:
		model(unsigned num_layers);
		~model();

		bool read_weights(const char *file, bool activation = true);
		
		unsigned char perf_forward_exec(m_data *img);
		unsigned char perf_forward_exec(float *input);

		void perf_fc_exec(fLayer *layer, float *img);
		void perf_act_exec(fLayer *layer);

	private:
		std::vector<fLayer *> m_layers;

		unsigned m_num_layers;
};

