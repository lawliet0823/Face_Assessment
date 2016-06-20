#include <iostream>
#include <fstream>
#include <map>
#include <math.h>
#include <cmath>
#include <opencv/cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "Headers/flandmark_detector.h"
#include "Headers/liblbp.h"

using namespace cv;
using namespace std;

// Preloading
const String face_cascade_name = "haarcascade_frontalface_alt_tree.xml";
FLANDMARK_Model *landmarkModel = flandmark_init("flandmark_model.dat");

//ShapeRegressor regressor;

map<string, int> readFile(string fileName);
vector<Rect> faceDetection(Mat image);
double* landmarkDetection(Mat image, Rect rect_face);
//vector<Mat_<int>> landmarkDetection_27(Mat &image, vector<Rect> faces);
Rect getCropImageBound(double center_x, double center_y, int crop_size);

double getSkewness(double *landmarks);
double caculateSymmetry(double *landmarks);
double caculateSymmetry(double *landmarks, double angle);
double caculateSymmetry(Mat image, double *landmarks);
double caculateSymmetry(Mat image, double *landmarks, Rect rect);
map<string, Mat> caculateSymmetry(map<string, Mat> infoMap, map<string, double*> landMap);
map<string, Mat> caculateSharpness(map<string, Mat> infoMap);
map<string, Mat> caculateBrightness(map<string, Mat> infoMap);
Mat rotateImage(Mat src, double angle);
double* landmarkRotation(double *landmarks, double angle, double midX, double midY);

int main() {

	ofstream assess_file("Assessment.txt");
	map<string, int> map_file = readFile("File.txt");
	map<string, int>::iterator it_map_file = map_file.begin();
	map<string, Mat> inputMap;
	map<string, double*> landMap;
	//vector<Mat> vm_img;
	//int previous_label = 4;
	int count = 1;
	for (; it_map_file != map_file.end(); it_map_file++) {
		string file_name = it_map_file->first;
		int label = it_map_file->second;

		//if (label == previous_label) {
			// Read image
		Mat face_image = imread(file_name);
		if (face_image.empty()) {
			cout << "Image read error" << file_name << endl;
			continue;
		}

		// Examine image size
		if (face_image.rows > 150 && face_image.cols > 150) {
			Mat grey_face_image;
			cvtColor(face_image, grey_face_image, CV_RGB2GRAY);

			// Face Detection
			vector<Rect> faces = faceDetection(face_image);
			if (faces.empty()) {
				cout << "face detection error " << file_name << endl;
				continue;
			}

			// Landmark detection
			double *landmarks = landmarkDetection(grey_face_image, faces.at(0));
			if (*landmarks < 0) {
				cout << "can't find landmarks " << file_name << endl;
				continue;
			}

			// Image rotation
			double angle = getSkewness(landmarks);
			Mat rotation_image = rotateImage(face_image, angle);
			
			
			vector<Mat> channels;
			Mat img_hist_equalized;
			cvtColor(rotation_image, img_hist_equalized, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format
			split(img_hist_equalized, channels); //split the image into channels
			equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)
			merge(channels, img_hist_equalized); //merge 3 channels including the modified 1st channel into one image
			cvtColor(img_hist_equalized, img_hist_equalized, CV_YCrCb2BGR); //change the color image from YCrCb to BGR format (to display image properly)
			//imshow("", img_hist_equalized);
			//waitKey(0);
			
			//inputMap.insert(make_pair(file_name, face_image));
			//landMap.insert(make_pair(file_name, landmarks));

			//cvtColor(rotation_image, rotation_image, CV_RGB2GRAY);
			cout << file_name << endl;
			//assess_file << count++ << "	" << caculateSymmetry(rotation_image, landmarkRotation(landmarks, -angle, face_image.cols / 2, face_image.rows / 2)) << endl;
			assess_file << count++ << "	" << caculateSymmetry(img_hist_equalized, landmarks) << endl;
			/*
			if (caculateSymmetry(rotation_image, landmarkRotation(landmarks, -angle, rotation_image.cols / 2, rotation_image.rows / 2)) < 120) {
				//tempMap.insert(make_pair(file_name, face_image));
				cout << file_name << endl;
				assess_file << file_name << endl;
			}
			*/
		}
		//}
			/*
		else {
			previous_label++;
			vector<Mat>().swap(vm_img);
		}
		*/
	}
	assess_file.close();
	/*
	map<string, Mat> finalMap = caculateSymmetry(inputMap, landMap);
	map<string, Mat>::iterator it_final = finalMap.begin();
	for (; it_final != finalMap.end(); it_final++) {
		assess_file << it_final->first << endl;
	}
	assess_file.close();
	*/

	/*
	ofstream assess_file("Assessment_File.txt");
	map<string, int> map_file = readFile("Before_Assess_Yeh.txt");
	map<string, int>::const_iterator it_map_file = map_file.begin();
	for (; it_map_file != map_file.end(); it_map_file++) {
		String fileLocation = it_map_file->first;
		Mat image = imread(fileLocation);
		vector<Rect> faces = faceDetection(image);
		if (faces.size() == 0) {
			cout << "Face Detect Error!!!" << endl;
		}
		Mat grey_image;
		cvtColor(image, grey_image, CV_RGB2GRAY);
		double *landmarks = landmarkDetection(grey_image, faces.at(0));
		if (*landmarks < 0) {
			cout << "Landmark Detection Failed" << endl;
		}
		Mat rotation_image = rotateImage(image, getSkewness(landmarks));
		double value = caculateSymmetry(rotation_image, landmarks);
		if (value < 140)
			cout << fileLocation << endl;
		assess_file << value << endl;
	}
	assess_file.close();
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
	double *landmarks = (double*)malloc(2 * landmarkModel->data.options.M * sizeof(double));
	int bbox[] = { rect_face.x,rect_face.y,rect_face.x + rect_face.width,rect_face.y + rect_face.height };
	flandmark_detect(img_grayscale, bbox, landmarkModel, landmarks);

	/*
	for (size_t landmark_element = 0; landmark_element < 15; landmark_element += 2) {
		cvCircle(img_grayscale, cvPoint(landmarks[landmark_element], landmarks[landmark_element + 1]), 3, CV_RGB(255, 0, 0), -1, 8, 0);
		cvShowImage("Result", img_grayscale);
		waitKey(500);
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
	return theta * 180 / 3.14;
}

double caculateSymmetry(double *landmarks) {
	float diff_value = 0;

	return diff_value;
}

double caculateSymmetry(double * landmarks, double angle)
{
	double score = 0;
	double la = 0;
	double lb = 0;
	double r = 0;
	if (angle < 5) {
		score = 1;
	}
	else {
		score = pow((1 - angle / 90), 2);
	}
	la = sqrt(pow(landmarks[4] - landmarks[2], 2) + pow(landmarks[5] - landmarks[3], 2));
	lb = sqrt(pow(landmarks[14] - (landmarks[2] + landmarks[4]) / 2, 2) + pow(landmarks[15] - (landmarks[3] + landmarks[5]) / 2, 2));
	r = lb / la;
	if (r < 1) {
		score = score*pow(r, 2);
	}
	return score;
}

/***
float caculateSymmetry(Mat image, double *landmarks, Rect rect) {

}
***/


double caculateSymmetry(Mat image, double *landmarks, Rect rect) {
	image(rect).copyTo(image);
	int crop_x = static_cast<int>(landmarks[0]) - rect.x;
	int crop_y = static_cast<int>(landmarks[1]) - rect.y;

	Rect rImageL(0, 0, crop_x, image.rows);
	Rect rImageR(crop_x, 0, image.cols - crop_x, image.rows);
	Mat imageL;
	Mat imageR;
	image(rImageL).copyTo(imageL);
	image(rImageR).copyTo(imageR);

	//imshow("L", imageL);
	//imshow("R", imageR);
	//waitKey(0);

	int hbins = 30, sbins = 32;
	int channels[] = { 0, 1 };
	int histSize[] = { hbins, sbins };
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 255 };
	//float vranges[] = { 0, 255 };
	const float *ranges[] = { hranges, sranges };

	Mat patch_HSV;
	Mat histL;
	Mat histR;
	cvtColor(imageL, patch_HSV, CV_BGR2HSV);
	calcHist(&patch_HSV, 1, channels, Mat(), histL, 2, histSize, ranges, true, false);
	normalize(histL, histL, 0, 1, CV_MINMAX);

	cvtColor(imageR, patch_HSV, CV_BGR2HSV);
	calcHist(&patch_HSV, 1, channels, Mat(), histR, 2, histSize, ranges, true, false);
	normalize(histR, histR, 0, 1, CV_MINMAX);

	int numrows = hbins*sbins;

	Mat signL = Mat(numrows, 3, CV_32FC1, Scalar::all(0));
	Mat signR = Mat(numrows, 3, CV_32FC1, Scalar::all(0));

	float value;
	for (size_t h = 0; h < hbins; h++) {
		for (size_t s = 0; s < sbins; s++) {
			//for (size_t v = 0; v < vbins; v++) {
			value = histL.at<float>(h, s);
			float *dataL = signL.ptr<float>(h*s);
			dataL[0] = value;
			dataL[1] = h;
			dataL[2] = s;
			//dataL[3] = v;
			//cout << h << "	" << s << "	" << v << endl;

			value = histR.at<float>(h, s);
			float *dataR = signR.ptr<float>(h*s);
			dataR[0] = value;
			dataR[1] = h;
			dataR[2] = s;
			//dataR[3] = v;
		//}
		}
	}
	/*
	int histSize = 255;	// bin number
	float range[] = { 0,255 };	// data range
	const float *histRange = { range };
	calcHist(&imageL, 1, 0, Mat(), histL, 1, &histSize, &histRange);
	calcHist(&imageR, 1, 0, Mat(), histR, 1, &histSize, &histRange);
	normalize(histL, histL, 0, 255, NORM_MINMAX);
	normalize(histR, histR, 0, 255, NORM_MINMAX);


	Mat signL(histL.rows, 2, CV_32FC1);
	Mat signR(histR.rows, 2, CV_32FC1);

	for (size_t i = 0; i < histL.rows; i++) {
		signL.at<float>(i, 0) = histL.at<float>(i, 0);
		signL.at<float>(i, 1) = i;
	}

	for (size_t i = 0; i < histR.rows; i++) {
		signR.at<float>(i, 0) = histR.at<float>(i, 0);
		signR.at<float>(i, 1) = i;
		//cout << histR.at<float>(i, 0) << endl;
	}

	//double value = compareHist(histL, histR, CV_COMP_INTERSECT);
	*/
	float emd_distance = EMD(signL, signR, CV_DIST_L2);
	return emd_distance;
}

// use hog to detect symmetry
double caculateSymmetry(Mat image, double *landmarks) {
	//cout << "in" << endl;
	double symmetryValue = 0;
	for (size_t landmark_element = 2; landmark_element < 13; landmark_element += 4) {
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
		HOGDescriptor hog(Size(32, 32), Size(16, 16), Size(8, 8), Size(4, 4), 9);
		//HOGDescriptor hogR(Size(32, 32), Size(16, 16), Size(8, 8), Size(8, 8), 9);
		vector<float> dersL;
		vector<float> dersR;
		vector<Point> locsL;
		vector<Point> locsR;
		//hog.compute(crop_ImgR, dersR, Size(8, 8), Size(0, 0));
		hog.compute(crop_ImgL, dersL, Size(4, 4), Size(0, 0), locsL);
		//cout << dersL.size() << "	" << dersR.size() << endl;
		hog.compute(crop_ImgR, dersR, Size(4, 4), Size(0, 0), locsR);
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
		//cout << "finish" << endl;
		symmetryValue += compareHist(matHistL, matHistR, CV_COMP_CHISQR);
	}
	symmetryValue /= 7;
	return symmetryValue;
}

map<string, Mat> caculateSymmetry(map<string, Mat> infoMap, map<string, double*> landMap)
{
	map<string, Mat> returnMap;
	map<float, string> mfs_score;
	map<string, Mat>::iterator it_info = infoMap.begin();
	for (; it_info != infoMap.end(); it_info++) {
		float score = 0;
		string file_name = it_info->first;
		Mat face_image = it_info->second;

		double angle = getSkewness(landMap.find(file_name)->second);
		Mat rotation_image = rotateImage(face_image, angle);
		it_info->second = rotation_image;
		score = caculateSymmetry(rotation_image, landmarkRotation(landMap.find(file_name)->second, -angle, rotation_image.cols / 2, rotation_image.rows / 2));
		mfs_score.insert(make_pair(score, file_name));
	}

	map<float, string>::reverse_iterator it_mfs = mfs_score.rbegin();
	for (size_t i = 0; i < 10; i++, it_mfs++) {
		returnMap.insert(make_pair(it_mfs->second, infoMap.find(it_mfs->second)->second));
	}
	return returnMap;
}

map<string, Mat> caculateSharpness(map<string, Mat> infoMap) {
	map<string, Mat> returnMap;
	map<float, string> mfs_score;
	//float maximum_score = INT_MIN;

	map<string, Mat>::iterator it_info = infoMap.begin();
	for (; it_info != infoMap.end(); it_info++) {
		string file_name = it_info->first;
		Mat face_image = it_info->second;
		Mat grey_image;
		cvtColor(face_image, grey_image, CV_RGB2GRAY);
		float score = 0;
		Mat low_pass_src;
		//IplImage *temp_image = &IplImage(src);
		//cvSmooth(temp_image, temp_image, CV_BLUR);
		//low_pass_src = Mat(temp_image);
		GaussianBlur(grey_image, low_pass_src, Size(3, 3), 0, 0);
		//imshow("Origin", src);
		//imshow("After", low_pass_src);
		//waitKey(0);
		for (size_t row_num = 0; row_num < grey_image.rows; row_num++) {
			for (size_t col_num = 0; col_num < grey_image.cols; col_num++) {
				score += abs(grey_image.at<uchar>(row_num, col_num) - low_pass_src.at<uchar>(row_num, col_num));
				//cout << "" << endl;
			}
		}
		score = score / (grey_image.rows*grey_image.cols);
		/*
		if (score > maximum_score) {
			maximum_score = score;
		}
		*/
		mfs_score.insert(make_pair(score, file_name));
	}
	map<float, string>::reverse_iterator it_mfs = mfs_score.rbegin();
	for (size_t i = 0; i < 10; i++, it_mfs++) {
		returnMap.insert(make_pair(it_mfs->second, infoMap.find(it_mfs->second)->second));
	}
	// Regulation depends on situation
	/*
	assert(images.size() == vf_img_score.size());
	for (size_t element_num = 0; element_num < vf_img_score.size(); element_num++) {
		vf_img_score[element_num] = vf_img_score[element_num] / maximum_score;
	}
	*/
	return returnMap;
}

map<string, Mat> caculateBrightness(map<string, Mat> infoMap) {
	map<string, Mat> returnMap;
	map<float, string> mfs_score;
	//float maximum_score = INT_MIN;

	map<string, Mat>::iterator it_info = infoMap.begin();
	for (; it_info != infoMap.end(); it_info++) {
		string file_name = it_info->first;
		Mat face_image = it_info->second;
		Mat grey_image;
		cvtColor(face_image, grey_image, CV_RGB2GRAY);
		float score = 0;
		for (size_t row_num = 0; row_num < grey_image.rows; row_num++) {
			for (size_t col_num = 0; col_num < grey_image.cols; col_num++) {
				score += grey_image.at<uchar>(row_num, col_num);
			}
		}
		score = score / (grey_image.rows*grey_image.cols);
		/*
		if (score > maximum_score) {
			maximum_score = score;
		}
		*/
		mfs_score.insert(make_pair(score, file_name));
	}
	map<float, string>::reverse_iterator it_mfs = mfs_score.rbegin();
	//assert(images.size() == vf_img_score.size());
	for (size_t i = 0; i < 10; i++, it_mfs++) {
		returnMap.insert(make_pair(it_mfs->second, infoMap.find(it_mfs->second)->second));
	}
	return returnMap;
}

Mat rotateImage(Mat src, double angle) {
	Mat dst;
	Point2f pt(src.cols / 2., src.rows / 2.);
	Mat r = getRotationMatrix2D(pt, -8, 1.0);
	warpAffine(src, dst, r, Size(src.cols, src.rows));
	return dst;
}

double *landmarkRotation(double *landmarks, double angle, double midX, double midY) {
	for (size_t i = 0; i < 15; i += 2) {
		//cout << landmarks[i] << "	" << landmarks[i + 1] << endl;
		double tempX = 0;
		double tempY = 0;
		landmarks[i] = landmarks[i] - midX;
		landmarks[i + 1] = landmarks[i + 1] - midY;
		tempX = (landmarks[i] * cos(angle*3.1415926 / 180) - landmarks[i + 1] * sin(angle*3.1415926 / 180)) + midX;
		tempY = (landmarks[i] * sin(angle*3.1415926 / 180) + landmarks[i + 1] * cos(angle*3.1415926 / 180)) + midY;
		//cout << tempX << "	" << tempY << endl;
		landmarks[i] = tempX;
		landmarks[i + 1] = tempY;
	}
	return landmarks;
}

/*
vector<Mat_<int>> landmarkDetection_27(Mat &image, vector<Rect> faces) {

	// Face Landmark detection pre-setup
	vector<BoundingBox> test_bounding_box;
	vector<Mat_<int>> landmark;

	for (size_t i = 0; i < faces.size(); i++) {
		// Face Detection visualization
		Point leftUpCorner(faces[i].x, faces[i].y);
		Point rightDownCorner(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
		//rectangle(image, leftUpCorner, rightDownCorner, Scalar(255, 0, 255), 4, 8, 0);

		// Face Detection Bound
		BoundingBox temp;
		temp.start_x = faces[i].x;
		temp.start_y = faces[i].y;
		temp.width = faces[i].width;
		temp.height = faces[i].height;
		temp.centroid_x = faces[i].x + faces[i].width / 2.0;
		temp.centroid_y = faces[i].y + faces[i].height / 2.0;
		test_bounding_box.push_back(temp);

		Mat_<int> current_shape = regressor.Predict(image, test_bounding_box.at(i), 30);
		landmark.push_back(current_shape);
		// Face image landmark visualization

		for (int j = 0; j < 27; j++) {
			//cout << current_shape(j, 0) << "		" << current_shape(j, 1) << endl;
			circle(image, Point2d(current_shape(j, 0), current_shape(j, 1)), 3, Scalar(255, 0, 0), -1, 8, 0);
		}

		imwrite("Result.jpg", image);
		//imshow("Face", image);
	}
	return landmark;
	//imshow(window_name,image);
}
*/
