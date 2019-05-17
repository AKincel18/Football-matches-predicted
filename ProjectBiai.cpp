#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <algorithm>

#include <stdint.h>
#include <limits.h>

// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;
using namespace std;

const int inputOutputCount = 21;
int inputCount = 18;
string title[inputOutputCount] = { "tableH", "tableA",
"pointsH", "pointsA", "goalsScoredHome","goalsScoredAway","shotsHome","shotsAway","onTargetshotsHome","onTargetshotsAway",
//C///////////D////////////E/////////////////F////////////////G///////////H//////////////I////////////////////J
"pointsH","pointsA","goalsScoredHome","goalsScoredAway","shotsHome","shotsAway","onTargetshotsHome","onTargetshotsAway",
////K/////////L/////////M///////////////////N/////////////////O//////////P///////////////Q////////////////R
"HWIN", "Draw", "AWIN" };

int hidden_layer_count = 10;

double training_instances_ratio = 0.7,
	   selection_instances_ratio = 0.15,
	   testing_instances_ratio = 0.15;
int main()
{
	srand((unsigned int)time(NULL));
	try {

		//Data set

		string path = "C://Users//Adam//Desktop//Projects//VS_2017//Neural Network//v23//";
		string path_to_input = "C://Users//Adam//Desktop//Projects//VS_2017//Neural Network//";
		DataSet data_set;
		data_set.set_data_file_name(path_to_input + "Inputs.csv");
		data_set.set_separator("Semicolon");
		data_set.load_data();

		//Variables
		Variables* variables_pointer = data_set.get_variables_pointer();

		for (int i = 0; i < inputOutputCount; i++)
		{
			variables_pointer->set_name(i, title[i]);
			if (i <= inputCount-1)
				variables_pointer->set_use(i, Variables::Input);
			else
				variables_pointer->set_use(i, Variables::Target);
		}
		const Matrix<std::string> inputs_information = variables_pointer->arrange_inputs_information();
		const Matrix<std::string> targets_information = variables_pointer->arrange_targets_information();

		std::cout << "Input information: " << std::endl << inputs_information << std::endl<<std::endl;
		std::cout << "Target information: " << std::endl << targets_information << std::endl << std::endl;

		//Instances
		//access to instances class, through the instances_pointer

		Instances* instances_pointer = data_set.get_instances_pointer();

		instances_pointer->split_random_indices(training_instances_ratio, selection_instances_ratio, testing_instances_ratio); //training_instances_ratio, selection_instances_ratio , testing_instances_ratio 
		
		const Vector<Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();

		std::cout << "/////////////////////////////////////////////////" << endl;
		std::cout << "Statistics:" << endl;
		std::cout << "/////////////////////////////////////////////////" << endl;
		for (int i = 0; i < inputCount; i++)
		{
			cout << title[i] << endl;
			cout << inputs_statistics[i] << endl;
		}


		

		// Neural network
		NeuralNetwork neural_network(inputCount, hidden_layer_count, inputOutputCount - inputCount); //inputs, number of neurons in the hidden layer of the multilayer perceptron, outputs

		Inputs* inputs_pointer = neural_network.get_inputs_pointer();

		inputs_pointer->set_information(inputs_information);

		Outputs* outputs_pointer = neural_network.get_outputs_pointer();

		outputs_pointer->set_information(targets_information);



		//SCALING DATA
		neural_network.construct_scaling_layer();

		ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

		scaling_layer_pointer->set_statistics(inputs_statistics);

		scaling_layer_pointer->set_scaling_method(ScalingLayer::NoScaling);

		neural_network.construct_probabilistic_layer();
		
		ProbabilisticLayer* probabilistic_layer_pointer = neural_network.get_probabilistic_layer_pointer();

		probabilistic_layer_pointer->set_probabilistic_method(ProbabilisticLayer::Probability);
		
		
		// Loss index
		LossIndex loss_index;

		loss_index.set_data_set_pointer(&data_set);
		loss_index.set_neural_network_pointer(&neural_network);

		//Training strategy
		TrainingStrategy training_strategy;

		training_strategy.set(&loss_index);



		//training algorithm
		training_strategy.set_main_type(TrainingStrategy::QUASI_NEWTON_METHOD);

		//modifiy some parameter in the training algorithm
		QuasiNewtonMethod * newton_method = training_strategy.get_quasi_Newton_method_pointer();
		
		newton_method->set_minimum_loss_increase(1.0e-6); // Sets a new minimum loss improvement during training.
		newton_method->set_loss_goal(1.0e-3); //Sets a new goal value for the loss. This is used as a stopping criterion when training a multilayer perceptron


		TrainingStrategy::Results training_strategy_results = training_strategy.perform_training();

		//Model selection
		ModelSelection model_selection(&training_strategy);

		//Sets a new method for selecting the inputs which have more impact on the targets.
		model_selection.set_inputs_selection_type(ModelSelection::GROWING_INPUTS);

		model_selection.set_order_selection_type(ModelSelection::SIMULATED_ANNEALING);


		//GrowingInputs* growing_inputs = model_selection.get_growing_inputs_pointer();

		//perform the inputs and the order selection
		model_selection.perform_inputs_selection();

		model_selection.perform_order_selection();



		///////////////////////////////////////////////////////////////////////
		////////////////////////Testing analysis///////////////////////////////
		///////////////////////////////////////////////////////////////////////
		
		TestingAnalysis testing_analysis(&neural_network, &data_set);
		
		TestingAnalysis::LinearRegressionResults linear_regression_results;
		linear_regression_results = testing_analysis.perform_linear_regression_analysis();
		for (size_t i = 0; i < linear_regression_results.linear_regression_parameters.size(); i++)
		{
			cout << "linear_regression_parameters for output " + to_string(i) << ": " << linear_regression_results.linear_regression_parameters[i] << endl;			
		}
		
		Matrix<size_t> confusion_matrix = testing_analysis.calculate_confusion();


		//Returns the results of a binary classification test in a single vector. 
		//http://www.opennn.net/documentation/reference/class_open_n_n_1_1_testing_analysis.html#a3682f04ac7793d00a53a7e0e7ccadd00
		
		Vector<double> binary_classification_tests = testing_analysis.calculate_binary_classification_tests();
		//calculate_multiple_classification_rates
		Matrix< Vector< size_t > > 	calculate_multiple_classification_rates = testing_analysis.calculate_multiple_classification_rates();
		



		//Returns a vector of matrices with number of rows equal to number of testing instances and number of columns equal to two (the targets value and the outputs value).
		//http://www.opennn.net/documentation/reference/class_open_n_n_1_1_testing_analysis.html#aec7519b6dd1af15e15a430245ea268cf

		Vector< Matrix<double> > target_output_data = testing_analysis.calculate_target_output_data();
		
		//NIEPEWNIAK
		//https://acutecaretesting.org/en/articles/roc-curves-what-are-they-and-how-are-they-used
		double calculate_area_under_curve = testing_analysis.calculate_area_under_curve(target_output_data[0], target_output_data[1]);

		//NIEPEWNIAK
		//Returns a matix with the values of a calibration plot. Number of columns is two. Number of rows varies depending on the distribution of positive targets.


		Matrix< double > calculate_calibration_plot = testing_analysis.calculate_calibration_plot(target_output_data[0], target_output_data[1]);

		




		Matrix<  size_t >  	calculate_confusion_multiple_classification = testing_analysis.calculate_confusion_multiple_classification(target_output_data[0], target_output_data[1]);


		double results_Home = 0.0, results_Draw = 0.0, results_Away = 0.0, all = 0.0, sum_all = 0.0;
		for (int i = 0; i < confusion_matrix.get_rows_number(); i++)
		{
			double sum = 0.0;
			for (int j = 0; j < confusion_matrix.get_columns_number(); j++)
			{
				sum += confusion_matrix(i, j);

			}
			switch (i)
			{
			case 0:
				results_Home = double(confusion_matrix(i, i)) / double(sum);
				break;
			case 1:
				results_Draw = double(confusion_matrix(i, i)) / double(sum);
				break;
			case 2:
				results_Away = double(confusion_matrix(i, i)) / double(sum);
				break;

			}
			sum_all += sum;
		}




		//Save results

		data_set.save(path + "data_set.xml");

		neural_network.save(path + "neural_network.xml");
		neural_network.save_expression(path + "expression.txt");

		training_strategy.save(path + "training_strategy.xml");
		training_strategy_results.save(path + "training_strategy_results.txt");

		model_selection.save(path + "model_selection.xml");
		//      model_selection_results.save("../data/model_selection_results.dat");

		confusion_matrix.save(path + "confusion.txt");

		calculate_multiple_classification_rates.save(path + "calculate_multiple_classification_rates.txt");
		
		//return the targets value and the outputs value
		target_output_data.save(path + "target_output_data.txt");

		calculate_calibration_plot.save(path + "calculate_calibration_plot.txt");

		calculate_confusion_multiple_classification.save(path + "calculate_multiple_classification_rates_params.txt");

		binary_classification_tests.save(path + "binary_classification_tests.txt");

		ofstream myfile(path + "results.txt");
		if (myfile.is_open())
		{
			myfile << results_Home << endl;
			myfile << results_Draw << endl;
			myfile << results_Away << endl;
			myfile << double(confusion_matrix(0, 0) + confusion_matrix(1, 1) + confusion_matrix(2,2))/double(sum_all) << endl;

			myfile << calculate_area_under_curve << endl;

			myfile.close();
		}

		ofstream myfile2(path + "ratio.txt");
		if (myfile2.is_open())
		{
			myfile2 << training_instances_ratio << " " <<
				selection_instances_ratio << " " <<
				testing_instances_ratio;


			myfile.close();
		}
		return 0;
	}
	catch (exception e)
	{
		cout << "Error: " << e.what() << endl;
		return 1;
	}
}