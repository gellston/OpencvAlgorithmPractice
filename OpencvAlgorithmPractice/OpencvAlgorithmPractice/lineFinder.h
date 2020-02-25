#pragma once

#define M_PI 3.14159265358979323846

#include <opencv2\opencv.hpp>

namespace oc {

	struct rect {
		double x;
		double y;
		double width;
		double height;
		double angle;

	};

	struct line {
		cv::Point2d center;
		double slope;
		double y_intercept;

		cv::Point2d start;
		cv::Point2d end;

		bool isVertical;
	};


	class lineFinder {

	public:

		lineFinder() {

		}

		void draw(cv::Mat _image, line _line, cv::Scalar _color = cv::Scalar(0, 255, 0), int _thickness = 2) {
			cv::line(_image, cv::Point(_line.start.x, _line.start.y), cv::Point(_line.end.x, _line.end.y), _color, _thickness);
		}

		void draw(cv::Mat _image, rect _rect, cv::Scalar _color = cv::Scalar(0, 255, 0), int _thickness = 2) {

			int stX = _rect.x;
			int endX = _rect.x + _rect.width;
			int stY = _rect.y;
			int endY = _rect.y + _rect.height;

			double angle = _rect.angle * M_PI / 180.0;

			int ltX = (stX - stX) * cos(angle) - (stY - stY)*sin(angle) + stX;
			int ltY = (stX - stX) * sin(angle) + (stY - stY)*cos(angle) + stY;

			int rtX = (endX - stX) * cos(angle) - (stY - stY)*sin(angle) + stX;
			int rtY = (endX - stX) * sin(angle) + (stY - stY)*cos(angle) + stY;

			int lbX = (stX - stX) * cos(angle) - (endY - stY)*sin(angle) + stX;
			int lbY = (stX - stX) * sin(angle) + (endY - stY)*cos(angle) + stY;

			int rbX = (endX - stX) * cos(angle) - (endY - stY)*sin(angle) + stX;
			int rbY = (endX - stX) * sin(angle) + (endY - stY)*cos(angle) + stY;

			cv::line(_image, cv::Point(ltX, ltY), cv::Point(rtX, rtY), _color, _thickness);
			cv::line(_image, cv::Point(rtX, rtY), cv::Point(rbX, rbY), _color, _thickness);
			cv::line(_image, cv::Point(rbX, rbY), cv::Point(lbX, lbY), _color, _thickness);
			cv::line(_image, cv::Point(lbX, lbY), cv::Point(ltX, ltY), _color, _thickness);
			
		}


		line compute(cv::Mat _image, rect _rect, double _diffThres, double _disThres, double _costThres) {
			std::vector<cv::Point2d> points;

			int stX = _rect.x;
			int endX = _rect.x + _rect.width;
			int stY = _rect.y;
			int endY = _rect.y + _rect.height;

			double angle = _rect.angle * M_PI / 180.0;

			for (int y = stY; y < endY; y++) {
				for (int x = stX; x < endX - 1; x++) {

					int currentAngleX = (x - stX) * cos(angle) - (y - stY)*sin(angle) + stX;
					int currentAngleY = (x - stX) * sin(angle) + (y - stY)*cos(angle) + stY;

					int nextAngleX = (x + 1 - stX) * cos(angle) - (y - stY)*sin(angle) + stX;
					int nextAngleY = (x + 1 - stX) * sin(angle) + (y - stY)*cos(angle) + stY;

					int currentX = currentAngleX;
					int currentY = currentAngleY;

					int nextX = nextAngleX;
					int nextY = nextAngleY;

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

			oc::line estimatedModel;
			oc::line finalModel;
			int pickSample = 2;
			std::vector<cv::Point2d> inlier;
			double max_cost = 0;


			int max_iteration = (int)(1 + log(1. - 0.99) / log(1. - pow(0.5, pickSample /* 횟수 */)));
			while (max_iteration-- && points.size() > pickSample) {
				std::vector<cv::Point2d>  ramdom = randomSample(points, pickSample);
				lineModelCompute(ramdom, estimatedModel, _image.cols, _image.rows);


				double cost = lineVerfication(inlier, estimatedModel, points, _disThres);


				if (cost >= max_cost) {
					max_cost = cost;
					finalModel = estimatedModel;
				}

			}

			return finalModel;
		}

	private:



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

		double lineDistance(line & _estimatedModel, cv::Point2d & _points) {

			double distance = abs(_estimatedModel.slope * _points.x - _points.y + _estimatedModel.y_intercept) / sqrt((_estimatedModel.slope * _estimatedModel.slope) + 1);
			//   | ax + by + c | / sqrt(a^2 + b^2) 직선과 점의 거리

			return distance;
		}

		double lineVerfication(std::vector<cv::Point2d> & _inlier, line & _estimatedModel, std::vector<cv::Point2d> & _points, double _distance_threshold)
		{
			_inlier.clear();

			double cost = 0.;

			for (int i = 0; i < _points.size(); i++) {
				// 직선에 내린 수선의 길이를 계산한다.
				double distance = lineDistance(_estimatedModel, _points[i]);
				
				// 예측된 모델에서 유효한 데이터인 경우, 유효한 데이터 집합에 더한다.
				if (distance < _distance_threshold) {
					//std::cout << "current Distance = " << distance << std::endl;
					cost += 1.;
					_inlier.push_back(_points[i]);
				}
			}
			return cost;
		}

		void lineModelCompute(std::vector<cv::Point2d> & _samples, line  & _model, double _width, double _height) {

			cv::Point2d center;



			center.x = (_samples[0].x + _samples[1].x) / 2;
			center.y = (_samples[0].y + _samples[1].y) / 2;
			_model.center = center;


			if (abs(_samples[0].x + _samples[1].x) == 0) {
				_model.isVertical = true;
				_model.slope = 0;
				_model.start.x = _model.center.x;
				_model.start.y = 0;

				_model.end.x = _model.center.y;
				_model.end.y = _height;
			}
			else {
				_model.isVertical = false;
				_model.slope = (_samples[0].y - _samples[1].y) / (_samples[0].x - _samples[1].x);
				//_model.slope = totalSlope;
				_model.y_intercept = center.y - _model.slope * center.x;

				_model.start.y = _model.slope * 0 + _model.y_intercept;
				_model.start.x = 0;

				_model.end.x = _width;
				_model.end.y = _model.slope * (_width) + _model.y_intercept;
			}

		}

	};

};