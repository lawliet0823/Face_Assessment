#include <iostream>
#include <fstream>
#include <map>
#include <math.h>
#include <opencv/cv.hpp>
#include <opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "Headers/flandmark_detector.h"
#include "Headers/liblbp.h"

using namespace cv;
using namespace std;

map<string, int> readFile(string fileName);
vector<Rect> faceDetection(Mat image);
double* landmarkDetection(Mat image, Rect rect_face);
Rect getCropImageBound(double center_x, double center_y, int crop_size);
double getSkewness(double *landmarks);
double caculateSymmetry(Mat image, double *landmarks, Rect rect);
vector<float> caculateSharpness(vector<Mat> vm_img);
vector<float> caculateBrightness(vector<Mat> vm_img);
Mat rotateImage(Mat src, double angle);

// Preloading
const String face_cascade_name = "haarcascade_frontalface_alt_tree.xml";
FLANDMARK_Model *landmarkModel = flandmark_init("flandmark_model.dat");

int main() {
	/*
	map<string, int> map_file = readFile("File_Location.txt");
	map<string, int>::iterator it_map_file = map_file.begin();
	for (; it_map_file != map_file.end(); it_map_file++) {
		string file_name = it_map_file->first;
		double label = it_map_file->second;

		Mat face_image = imread(file_name, 0);
		cout << file_name << endl;
		if (face_image.empty()) {
			cout << "read error" << endl;
			continue;
		}

		vector<Rect> faces = faceDetection(face_image);
		double *landmarks = landmarkDetection(face_image, faces[0]);
		if (*landmarks < 0) {
			cout << "can't find images " << file_name << endl;
			continue;
		}
		cout << caculateSymmetry(face_image, landmarks) << file_name << endl;
	}
	*/

	Mat image = imread("Crop_Image//009//018.jpg", 0);
	//equalizeHist(image, image);
	//resize(image, image, Size(250, 250));
	if (image.empty())
		cout << "can't read" << endl;
	//Mat dst = rotateImage(image, -5);
	Mat dst = image;
	vector<Rect> faces = faceDetection(dst);
	if (faces.size() == 0) {
		cout << "Can't detect face" << endl;
		waitKey(0);
	}
	//double *landmarks = landmarkDetection(image, faces[0]);
	double *landmarks = landmarkDetection(dst, faces[0]);
	if (*landmarks < 0)
		cout << "fail" << endl;
	cout << caculateSymmetry(image, landmarks, faces[0]) << endl;
	/*
	HOGDescriptor hog;
	vector<float> ders;
	vector<Point> locs;
	hog.compute(image, ders, Size(10, 10), Size(0, 0), locs);
	cout << ders.size() << endl;
	*/

	system("pause");
	return 0;
}

map<string, int> readFile(string fileName) {
	map<string, int> fileMap;
	string lineString;
	fstream fileReader(fileName);
	while (getline(fileReader, lineString)) {
		istringstream isstream(lineString);
		string file_Location;
		int label;
		isstream >> file_Location >> label;
		//cout << file_Location << "	" << label << endl;
		fileMap.insert(make_pair(file_Location, label));
	}
	return fileMap;
}

// Face Detection: return face bound
vector<Rect> faceDetection(Mat image) {
	CascadeClassifier face_cascade;
	vector<Rect> faces;

	if (!face_cascade.load(face_cascade_name)) {
		printf("Error Loading");
	}

	face_cascade.detectMultiScale(image, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cvSize(90, 90));
	return faces;
}

// flandmark version
double* landmarkDetection(Mat image, Rect rect_face) {
	IplImage *img_grayscale = &IplImage(image);
	//IplImage *img_grayscale = cvCreateImage(cvSize(i_image->width, i_image->height), IPL_DEPTH_8U, 1);
	//cvCvtColor(i_image,img_grayscale,CV_BGR2GRAY);
	double *landmarks = (double*)malloc(2 * landmarkModel->data.options.M*sizeof(double));
	int bbox[] = { rect_face.x,rect_face.y,rect_face.x + rect_face.width,rect_face.y + rect_face.height };
	flandmark_detect(img_grayscale, bbox, landmarkModel, landmarks);

	/*
	for (size_t landmark_element = 2; landmark_element < 15; landmark_element += 2) {
		cvCircle(img_grayscale, cvPoint(landmarks[landmark_element], landmarks[landmark_element + 1]), 3, CV_RGB(255, 255, 0), 1, 8, 3);
		cout << landmarks[landmark_element] << "	" << landmarks[landmark_element + 1] << endl;
	}
	*/
	return landmarks;
}

Rect getCropImageBound(double center_x, double center_y, int crop_size) {
	Rect rect(static_cast<int>(center_x) - crop_size / 2, static_cast<int>(center_y) - crop_size / 2, crop_size, crop_size);
	return rect;
}

double getSkewness(double *landmarks) {
	double mean_x_value = (landmarks[2] + landmarks[4] + landmarks[10] + landmarks[12]) / 4.0;
	double mean_y_value = (landmarks[3] + landmarks[5] + landmarks[11] + landmarks[13]) / 4.0;
	double theta = atan(((landmarks[2] * landmarks[3] + landmarks[4] * landmarks[5] + landmarks[10] * landmarks[11] + landmarks[12] * landmarks[13]) - 4 * mean_x_value*mean_y_value)
		/ (pow(landmarks[2], 2) + pow(landmarks[4], 2) + pow(landmarks[10], 2) + pow(landmarks[12], 2) - 4 * pow(mean_x_value, 2)));
	return theta * 180 / 3.1415926;
}

double caculateSymmetry(Mat image, double *landmarks, Rect rect) {
	image = image(rect);
	int crop_x = static_cast<int>(landmarks[0]) - rect.x;
	int crop_y = static_cast<int>(landmarks[1]) - rect.y;
	Rect rImageL(0, 0, crop_x, image.rows);
	Rect rImageR(crop_x, 0, image.cols - crop_x, image.rows);
	Mat imageL = image(rImageL);
	Mat imageR = image(rImageR);

	//imshow("L", imageL);
	//imshow("R", imageR);
	//waitKey(0);

	Mat histL;
	Mat histR;
	int histSize = 255;	// bin number
	float range[] = { 0,255 };	// data range
	const float *histRange = { range };
	calcHist(&imageL, 1, 0, Mat(), histL, 1, &histSize, &histRange);
	calcHist(&imageR, 1, 0, Mat(), histR, 1, &histSize, &histRange);
	normalize(histL, histL, 0, 255, CV_MINMAX);
	normalize(histR, histR, 0, 255, CV_MINMAX);

	Mat signL(histL.rows, 2, CV_32F);
	Mat signR(histR.rows, 2, CV_32F);

	for (size_t i = 0; i < histL.rows; i++) {
		signL.at<float>(i, 0) = i;
		signL.at<float>(i, 1) = histL.at<float>(i, 0);
	}

	for (size_t i = 0; i < histR.rows; i++) {
		signR.at<float>(i, 0) = i;
		signR.at<float>(i, 1) = histR.at<float>(i, 0);
	}

	//double value = compareHist(histL, histR, CV_COMP_INTERSECT);
	double value = EMD(signL, signR, CV_DIST_L1);
	return value;
}


/*
// use hog to detect symmetry
double caculateSymmetry(Mat image, double *landmarks) {
	//cout << "in" << endl;
	double symmetryValue = 0;
	for (size_t landmark_element = 2; landmark_element < 15; landmark_element += 4) {
		Rect cropRectL = getCropImageBound(static_cast<int>(landmarks[landmark_element]),
			static_cast<int>(landmarks[landmark_element + 1]), 32);
		Rect cropRectR = getCropImageBound(static_cast<int>(landmarks[landmark_element + 2]),
			static_cast<int>(landmarks[landmark_element + 3]), 32);
		Mat crop_ImgL = image(cropRectL);
		Mat crop_ImgR = image(cropRectR);
		//imshow("image", image);
		//imshow("L", crop_ImgL);
		//imshow("R", crop_ImgR);
		//waitKey(0);

		// HOG Descriptor
		HOGDescriptor hog(Size(32, 32), Size(16, 16), Size(8, 8), Size(8, 8), 9);
		//HOGDescriptor hogR(Size(32, 32), Size(16, 16), Size(8, 8), Size(8, 8), 9);
		vector<float> dersL;
		vector<float> dersR;
		vector<Point> locsL;
		vector<Point> locsR;
		//hog.compute(crop_ImgR, dersR, Size(8, 8), Size(0, 0));
		hog.compute(crop_ImgL, dersL, Size(8, 8), Size(0, 0), locsL);
		//cout << dersL.size() << "	" << dersR.size() << endl;
		hog.compute(crop_ImgR, dersR, Size(8, 8), Size(0, 0), locsR);
		//cout << dersL.size() << "	" << dersR.size() << endl;
		Mat hogFeatL;
		Mat hogFeatR;
		hogFeatL.create(dersL.size(), 1, CV_32FC1);
		hogFeatR.create(dersR.size(), 1, CV_32FC1);

		for (size_t i = 0; i < dersL.size(); i++) {
			hogFeatL.at<float>(i, 0) = dersL.at(i);
		}
		for (size_t i = 0; i < dersR.size(); i++) {
			hogFeatR.at<float>(i, 0) = dersR.at(i);
		}

		int nHistSize = 65536;
		float fRange[] = { 0.00f, 1.00f };
		const float* fHistRange = { fRange };
		Mat matHistL;
		Mat matHistR;
		calcHist(&hogFeatL, 1, 0, Mat(), matHistL, 1, &nHistSize, &fHistRange);
		calcHist(&hogFeatR, 1, 0, Mat(), matHistR, 1, &nHistSize, &fHistRange);
		//crop_ImgL.release();
		//crop_ImgR.release();
		//hogFeatL.release();
		//hogFeatR.release();

		cout << "finish" << endl;
		symmetryValue += compareHist(matHistL, matHistR, CV_COMP_INTERSECT);
	}
	symmetryValue /= 7;
	return symmetryValue;
}
*/

vector<float> caculateSharpness(vector<Mat> images) {
	assert(images.size() > 0);
	vector<float> vf_img_score;
	float maximum_score = INT_MIN;
	for (size_t element_num = 0; element_num < images.size(); element_num++) {
		float score = 0;
		Mat src = images.at(element_num);
		assert(src.channels() == 1);
		Mat low_pass_src;
		//IplImage *temp_image = &IplImage(src);
		//cvSmooth(temp_image, temp_image, CV_BLUR);
		//low_pass_src = Mat(temp_image);
		GaussianBlur(src, low_pass_src, Size(3, 3), 0, 0);
		//imshow("Origin", src);
		//imshow("After", low_pass_src);
		//waitKey(0);
		for (size_t row_num = 0; row_num < src.rows; row_num++) {
			for (size_t col_num = 0; col_num < src.cols; col_num++) {
				score += abs(src.at<uchar>(row_num, col_num) - low_pass_src.at<uchar>(row_num, col_num));
				//cout << "" << endl;
			}
		}
		score = score / (src.rows*src.cols);
		if (score > maximum_score) {
			maximum_score = score;
		}
		vf_img_score.push_back(score);
	}

	assert(images.size() == vf_img_score.size());
	for (size_t element_num = 0; element_num < vf_img_score.size(); element_num++) {
		vf_img_score[element_num] = vf_img_score[element_num] / maximum_score;
	}
	return vf_img_score;
}

vector<float> caculateBrightness(vector<Mat> images) {
	assert(images.size() > 0);
	vector<float> vf_img_score;
	float maximum_score = INT_MIN;
	for (size_t element_num = 0; element_num < images.size(); element_num++) {
		float score = 0;
		Mat src = images.at(element_num);
		for (size_t row_num = 0; row_num < src.rows; row_num++) {
			for (size_t col_num = 0; col_num < src.cols; col_num++) {
				score += src.at<uchar>(row_num, col_num);
			}
		}
		score = score / (src.rows*src.cols);
		if (score > maximum_score) {
			maximum_score = score;
		}
		vf_img_score.push_back(score);
	}

	assert(images.size() == vf_img_score.size());
	for (size_t element_num = 0; element_num < vf_img_score.size(); element_num++) {
		vf_img_score[element_num] = vf_img_score[element_num] / maximum_score;
	}
	return vf_img_score;
}

Mat rotateImage(Mat src, double angle) {
	Mat dst;
	Point2f pt(src.cols / 2., src.rows / 2.);
	Mat r = getRotationMatrix2D(pt, -8, 1.0);
	warpAffine(src, dst, r, Size(src.cols, src.rows));
	return dst;
}
