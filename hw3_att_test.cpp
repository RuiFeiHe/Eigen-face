#include "hw3_att_face.h"
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

int main(int argc, char** argv) {

	//load faces
	Face_ATT facelib;
	facelib.load("D:\\codes\\cv\\resources\\dataset\\att");

	//load model
	string model_name = "eigen_att.model";
	string file_name = "D:\\codes\\cv\\resources\\dataset\\att\\41\\05";
	if (argc >= 3) {
		model_name = argv[2];
		file_name = argv[1];
	}
	FileStorage model(model_name, FileStorage::READ);
	Mat e_vector_mat, e_value_mat, mean;
	model["e_vector_mat"] >> e_vector_mat;
	model["e_value_mat"] >> e_value_mat;
	model["mean"] >> mean;

	//rebuild
	Mat samples;
	Per_Face_ATT face;
	face.load(file_name);
	facelib.samples.copyTo(samples);
	
	Mat distance;
	distance = e_vector_mat * samples;
	Mat face_vect = e_vector_mat * face.vect; // kxMN x MNx1

	for (int i = 0; i < e_vector_mat.rows; i++) {
		normalize(e_vector_mat.row(i), e_vector_mat.row(i), 1.0, 0.0, NORM_L2);
	}
	Mat face_re = e_vector_mat * (face.vect - mean); // kxMN x MNx1
	Mat rebuild = e_vector_mat.t() * face_re + mean;  // MNxk x kx1 
	transpose(rebuild, rebuild);
	transpose(mean, mean);
	Mat rebuild_img = toImg(rebuild, WIDTH, HEIGHT);
	Mat mean_img = toImg(mean, WIDTH, HEIGHT);
	rebuild_img.convertTo(rebuild_img, CV_8U, 255);
	mean_img.convertTo(mean_img, CV_8U, 255);
	
	// find similar pic
	double min_d = norm(face_vect, distance.col(0), NORM_L2);
	double temp_d = 0;
	int min_i = 0;

	for (int i = 1; i < distance.cols; ++i) {
		temp_d = norm(face_vect, distance.col(i), NORM_L2);
		if (temp_d <= min_d) {
			min_d = temp_d;
			min_i = i;
		}
	}
	cout << (min_i / 5) + 1 << "/" << (min_i % 5) + 1 << " " << endl;
	Mat origin_mat = face.gray_pic;
	Mat similar_mat = facelib.faces.at(min_i)->gray_pic;
	string text = to_string(min_i / 5 + 1) + " No." + to_string(min_i % 5 + 1);
	cout << text << endl;
	putText(origin_mat, text, Point(10, 20), FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2, 8);

	//display
	namedWindow("Ori Pic", 0);
	namedWindow("Similar Pic", 0);
	namedWindow("rebuild face", 0);
	namedWindow("mean face", 0);
	cvResizeWindow("Ori Pic", 300, 300);
	cvResizeWindow("Similar Pic", 300, 300);
	cvResizeWindow("rebuild face", 300, 300);
	cvResizeWindow("mean face", 300, 300);
	imshow("Ori Pic", origin_mat);
	imshow("Similar Pic", similar_mat);
	imshow("rebuild face", rebuild_img);
	imshow("mean face", mean_img);
	waitKey(0);
	destroyAllWindows();
	return 0;
}