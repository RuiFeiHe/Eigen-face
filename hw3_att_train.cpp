#include "hw3_att_face.h"
using namespace std;
using namespace cv;
const int WIDTH = 92;
const int HEIGHT = 112;

Mat toImg(Mat vect, int w, int h) {
	assert(vect.type() == 6);
	assert(vect.cols == w * h);

	Mat result(Size(w, h), CV_64FC1);
	for (int i = 0; i < h; ++i) {
		vect.colRange(i * w, (i + 1) * w).convertTo(result.row(i), CV_64FC1);
	}
	normalize(result, result, 1.0, 0.0, NORM_MINMAX);
	return result;
}

void cal_mean(Mat samples, Mat& mean_save)
{					          // 1xMN
	// calculate mean face
	for (int i = 0; i < samples.rows; i++) {
		mean_save.row(i) = cv::mean(samples.row(i));
	}

}

void cal_Cov(Mat samples, Mat& cov)
{								
	// calculate covar matrix
	int k = samples.cols;
	Mat A;
	A.create(samples.rows, k, CV_64FC1);
	A = samples.col(0);
	for (int i = 1; i < samples.cols; i++)
	{
		hconcat(A, samples.col(i), A);
	}
	cov = A.t()*A;
}

int main(int argc, char** argv) {
	string model_name = "eigen_att.model";
	double energy = 0.95;
	if (argc >= 3) {
		model_name = argv[2];
		energy = atof(argv[1]);
	}

	//load
	Face_ATT facelib;
	facelib.load("D:\\codes\\cv\\resources\\dataset\\orl");
	Mat samples, cov_mat,mean_save;
	facelib.samples.copyTo(samples);
	mean_save.create(samples.rows, 1, CV_64FC1);
	
	// Cal Mean-face and Covar Matrix
	cal_mean(samples, mean_save);
	for (int i = 0; i < samples.cols; ++i) {
		samples.col(i) -= mean_save;
	}
	cout << "Calculating Mean Face and Covariance Mat..." << endl;
	cal_Cov(samples, cov_mat);
	cov_mat = cov_mat / (samples.rows - 1);

	Mat e_vector_mat, e_value_mat;
	cout << "Calculating Eigen value and vector..." << endl;
	eigen(cov_mat, e_value_mat, e_vector_mat);

	// Choosing number of Eigen vectors based on energy calculation
	double value_sum = sum(e_value_mat)[0];
	double energy_level = value_sum * energy;
	double energy_sum = 0;
	int k = 0;
	for (k = 0; k < e_value_mat.rows; k++)
	{
		energy_sum += e_value_mat.at<double>(k, 0);
		if (energy_sum >= energy_level) break;
	}
	e_vector_mat = (samples * e_vector_mat.t()).t();
	e_vector_mat = e_vector_mat.rowRange(0, k);
	e_value_mat = e_value_mat.rowRange(0, k);

	//store model
	cout << "Storing model..." << endl;
	FileStorage model(model_name, FileStorage::WRITE);
	model << "e_vector_mat" << e_vector_mat;
	model << "e_value_mat" << e_value_mat;
	model << "mean" << mean_save;
	model.release();

	//display
	vector<Mat> Top10EigenFace;
	for (int i = 0; i < 10; ++i) {
		Top10EigenFace.push_back(toImg(e_vector_mat.row(i) + mean_save.t(), WIDTH, HEIGHT));
		//Combine += 0.1 * e_vector_mat.row(i);
	}
	Mat meanface = toImg(mean_save.t(), WIDTH, HEIGHT);
	meanface.convertTo(meanface, CV_8U, 255);

	Mat result;
	hconcat(Top10EigenFace, result);
	result.convertTo(result, CV_8U, 255);

	imshow("Top10EigenFace", result);
	imshow("Combined Face", meanface);
	imwrite("Top10EigenFace.png", result);
	waitKey(0);

	destroyAllWindows();
	return 0;
}

