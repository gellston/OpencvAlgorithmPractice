#pragma once

#define M_PI 3.14159265358979323846

#include <opencv2\opencv.hpp>

namespace oc {

	/// circle 구조체
	struct circle {
		double x;
		double y;
		double r;
	};

	class circleFinder {

	private:
	
		/// circle과 점간에 거리를 측정하는 함수
		double circleDistance(circle _circle, cv::Point2d _point)	
		{
			double dx = _circle.x - _point.x;
			double dy = _circle.y - _point.y;
			return fabs(sqrt(dx*dx + dy * dy) - _circle.r);
		}

		///랜덤하게 뽑은 점의 중복 여부를 확인하는 함수
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

		/// 주어진 갯수만큼 inlierPoint에서 포인트를 뽑는 함수.
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

		/// circle 모델을 구하는 함수. (최소자승법)
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
		

		/// circle 모델을 검증하는 함수.
		double circleVerfication(std::vector<cv::Point2d> & _inliear, circle & _estimatedModel, std::vector<cv::Point2d> & _points, double _distance_threshold)
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



	public:

		circleFinder() {

		}

		void draw(cv::Mat _image, circle centerPoint, cv::Scalar _color = cv::Scalar(0, 255, 0), int _thickness = 2) {
			cv::circle(_image, cv::Point(centerPoint.x, centerPoint.y), centerPoint.r, _color, _thickness);
		}

		circle compute(cv::Mat _image, circle centerPoint, double _minRad, double _maxRad, double _stepAng, int _diffThres, int _disThres, double _costThres=0.9) {

			std::vector<cv::Point2d> inlierPoints;

			std::vector<cv::Point2d> points;
			for (double angle = 0; angle < 360; angle += _stepAng) {
				for (int radius = _minRad; radius < _maxRad; radius++) {

					double degree = angle * M_PI / 180.0;

					int currentX = (int)centerPoint.x + cos(degree) * radius;
					int currentY = (int)centerPoint.y + sin(degree) * radius;

					int nextX = (int)centerPoint.x + cos(degree) * (radius + 1);
					int nextY = (int)centerPoint.y + sin(degree) * (radius + 1);

					if (currentX >= _image.cols || currentX < 0) continue;
					if (currentY >= _image.rows || currentY < 0) continue;
					if (nextX >= _image.cols || nextX < 0) continue;
					if (nextY >= _image.rows || nextY < 0) continue;

					unsigned char * imgPointer = static_cast<unsigned char *>(_image.data);

					double currentPixelValue = imgPointer[currentY *  _image.cols + currentX];
					double nextPixelValue = imgPointer[nextY *  _image.cols + nextX];

					if (abs(currentPixelValue - nextPixelValue) > _diffThres) {
						points.push_back(cv::Point2d(currentX, currentY));
						break;
					}
				}
			}

			circle estimated_model;
			circle final_model;

			int sampleCount = 3;
			int maxCost = 0;

			int max_iteration = (int)(1 + log(1. - 0.99) / log(1. - pow(0.5, sampleCount /* 횟수 */)));

			for (int index = 0; index < max_iteration; index++) {
				std::vector<cv::Point2d> randomPoints = randomSample(points, sampleCount);					// random 포인트 추출
				circleModelCompute(randomPoints, estimated_model);												//circle 모델 생성
				double cost = circleVerfication(inlierPoints, estimated_model, points, _disThres);

				if (maxCost < cost) {
					maxCost = cost;

					circleModelCompute(inlierPoints, final_model);
					//if (maxCost / points.size() > _costThres)
					//	break;
				}
			}

			return final_model;
		}
	};
};
