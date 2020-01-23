// OpencvAlgorithmPractice.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <opencv2\opencv.hpp>
#include <Windows.h>

struct circle {
	double x;
	double y;
	double r;
};

double circleDistance(circle _circle, cv::Point2d _point)
{
	double dx = _circle.x - _point.x;
	double dy = _circle.y - _point.y;
	return fabs(sqrt(dx*dx + dy * dy) - _circle.r);
}

bool checkRamdomOverlap(cv::Point2d _selectedPoint, std::vector<cv::Point2d> & _savedPoints)
{
	for (int i = 0; i < _savedPoints.size(); i++) {
		if (_selectedPoint.x == _savedPoints[i].x && _selectedPoint.y == _savedPoints[i].y) 
		{
			return true;
		}
	}
	return false;
}

std::vector<cv::Point2d> randomSample(std::vector<cv::Point2d> & _samples, int _sample_count) 
{
	std::vector<cv::Point2d> savedPoints;
	int sampleSize = _samples.size();
	for (int index = 0; index < _sample_count;) {

		int generatedIndex = rand() % sampleSize;
		cv::Point2d selectedPoint = _samples[generatedIndex];

		if (!checkRamdomOverlap(selectedPoint, savedPoints)) {
			savedPoints.push_back(selectedPoint);
			++index;
		}
	};

	return savedPoints;
}

void circleModelCompute(std::vector<cv::Point2d> & _samples, circle  & _model) {
	cv::Mat A = cv::Mat::ones(_samples.size(), 3, CV_64F);
	cv::Mat B = cv::Mat::ones(_samples.size(), 1, CV_64F);

	for (int index = 0; index < _samples.size(); index++) {
		double x = _samples[index].x;
		double y = _samples[index].y;
		A.at<double>(index, 0) = x;
		A.at<double>(index, 1) = y;

		B.at<double>(index, 0) = -x * x - y * y;
	}

	cv::Mat tranA = A.t();
	cv::Mat invA = (tranA * A).inv() * tranA;
	cv::Mat x = invA * B;

	double cx = -x.at<double>(0, 0) / 2.0;
	double cy = -x.at<double>(1, 0) / 2.0;
	double r = sqrt(cx*cx + cy * cy - x.at<double>(2, 0));

	_model.x = cx;
	_model.y = cy;
	_model.r = r;
}


double circleVerfication(std::vector<cv::Point2d> & _inliear, 
						circle & _estimatedModel, 
						std::vector<cv::Point2d> & _points,
						double _distance_threshold)
{
	_inliear.clear();

	double cost = 0.;

	for (int i = 0; i < _points.size(); i++) {
		// 직선에 내린 수선의 길이를 계산한다.
		double distance = circleDistance(_estimatedModel, _points[i]);

		// 예측된 모델에서 유효한 데이터인 경우, 유효한 데이터 집합에 더한다.
		if (distance < _distance_threshold) {
			cost += 1.;
			_inliear.push_back(_points[i]);
		}
	}
	return cost;
}

int main()
{

	LARGE_INTEGER frequency;
	LARGE_INTEGER beginTime;
	LARGE_INTEGER endTime;


	cv::Mat image = cv::imread("C://Github//OpencvAlgorithmPractice//OpencvAlgorithmPractice//images//circle.png", cv::IMREAD_GRAYSCALE);
	cv::Mat color = cv::imread("C://Github//OpencvAlgorithmPractice//OpencvAlgorithmPractice//images//circle.png", cv::IMREAD_COLOR);


	//프로그램이나 클래스 시작부분에
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&beginTime);


	int minRadius = 100;
	int maxRadius = 200;
	int samplingCount = 3;
	double stepAngle = 15;
	double pointThreshold = 50;
	double distanceThreshold = 10;
	cv::Point2d center;
	center.x = 230;
	center.y = 200;


	std::vector<cv::Point2d> inlierPoints;
	circle estimated_model;
	circle final_model;
	double maxCost = 0;


	std::vector<cv::Point2d> points;
	for (double angle = 0; angle < 360; angle += stepAngle) {
		for (int radius = minRadius; radius < maxRadius; radius++) {
			int currentX = (int)center.x + cos(angle) * radius;
			int currentY = (int)center.y + sin(angle) * radius;

			int nextX = (int)center.x + cos(angle) * (radius + 1);
			int nextY = (int)center.y + sin(angle) * (radius + 1);

			if (currentX >= image.cols || currentX < 0) continue;
			if (currentY >= image.rows || currentY < 0) continue;
			if (nextX >= image.cols || nextX < 0) continue;
			if (nextY >= image.rows || nextY < 0) continue;

			unsigned char * imgPointer = static_cast<unsigned char *>(image.data);
			
			double currentPixelValue = imgPointer[currentY *  image.cols + currentX];
			double nextPixelValue = imgPointer[nextY *  image.cols + nextX];

			if (abs(currentPixelValue - nextPixelValue) > pointThreshold) {
				points.push_back(cv::Point2d(currentX, currentY));
				break;
			}
		}
	}


	////////////////////// Ransac
	int max_iteration = (int)(1 + log(1. - 0.99) / log(1. - pow(0.5, samplingCount /* 횟수 */ )));

	for (int index = 0; index < max_iteration; index++) {
		std::vector<cv::Point2d> randomPoints = randomSample(points, samplingCount);					// random 포인트 추출
		circleModelCompute(randomPoints, estimated_model);											//circle 모델 생성
		double cost = circleVerfication(inlierPoints, estimated_model, points, distanceThreshold);

		if (maxCost < cost) {
			maxCost = cost;
			
			circleModelCompute(inlierPoints, final_model);
			if (maxCost / points.size() > 0.8) 
				break;
		}
	}


	QueryPerformanceCounter(&endTime);
	int64 elapsed = endTime.QuadPart - beginTime.QuadPart;
	double taktTime = (double)elapsed / (double)frequency.QuadPart * 1000;
	std::cout << "inference Time = " << taktTime << std::endl;

	for (int index = 0; index < points.size(); index++) {
		cv::circle(color, points[index], 5, cv::Scalar(255, 0, 0), -1);
	}

	cv::circle(color, cv::Point(final_model.x, final_model.y), final_model.r, cv::Scalar(0, 0, 255), 2);



	cv::imshow("original", image);
	cv::imshow("color", color);
	cv::waitKey();
}
