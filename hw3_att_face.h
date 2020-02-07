#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;


class Per_Face_ATT {
public:
	Mat origin_pic;
	Mat gray_pic;
	Mat transformed_pic;
	int x1, y1, x2, y2;
	Mat trans_mat;
	Mat_<double> equalized_mat;
	Mat_<double> vect;
	void load(String path) {
		load_eye_pos(path);
		path += ".png";
		origin_pic = imread(path);
		gray_pic = imread(path, IMREAD_GRAYSCALE);
		//cout << "load" <<path<<gray_pic.size()<< endl;
		transform();
	}

	void load_eye_pos(String path) {
		path += ".txt";
		ifstream file(path, ifstream::in);
		file >> x1 >> y1 >> x2 >> y2;
	}

	void transform() {
		Point center((x1 + x2) / 2, (y1 + y2) / 2);
		double angle = atan((double)(y2 - y1) / (double)(x2 - x1)) * 180.0 / CV_PI;
		trans_mat = getRotationMatrix2D(center, angle, 1.0);
		trans_mat.at<double>(0, 2) += 45 - center.x;
		trans_mat.at<double>(1, 2) += 56 - center.y;
		warpAffine(gray_pic, transformed_pic, trans_mat, gray_pic.size());
		equalizeHist(transformed_pic, transformed_pic);
		transformed_pic.copyTo(equalized_mat);
		vect = equalized_mat.reshape(1, 1).t();
	}
};

class Face_ATT {
public:
	int num_of_persons = 41;
	int faces_per_person = 5;
	vector<Per_Face_ATT*> faces;
	vector<Mat_<double>> _samples;
	Mat_<double> samples;

	void load(string path) {
		for (int i = 1; i <= num_of_persons; i++)
		{
			for (int j = 1; j <= faces_per_person; ++j) {
				string per_path = path + "\\" + to_string(i) + "\\0" + to_string(j);
				Per_Face_ATT* face = new Per_Face_ATT();
				face->load(per_path);
				faces.push_back(face);
				_samples.push_back(face->vect);
			}
		}
		hconcat(_samples, samples);
	}

};