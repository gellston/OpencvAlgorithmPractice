// OpencvAlgorithmPractice.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//
#include <iostream>
#include <opencv2\opencv.hpp>
#include <Windows.h>


#include "circleFinder.h"
#include "lineFinder.h"



int main()
{

	LARGE_INTEGER frequency;
	LARGE_INTEGER beginTime;
	LARGE_INTEGER endTime;


	cv::Mat image1 = cv::imread("C://Github//OpencvAlgorithmPractice//OpencvAlgorithmPractice//images//circle.png", cv::IMREAD_GRAYSCALE);
	cv::Mat color1 = cv::imread("C://Github//OpencvAlgorithmPractice//OpencvAlgorithmPractice//images//circle.png", cv::IMREAD_COLOR);


	cv::Mat image2 = cv::imread("C://Github//OpencvAlgorithmPractice//OpencvAlgorithmPractice//images//blackandwhite.png", cv::IMREAD_GRAYSCALE);
	cv::Mat color2 = cv::imread("C://Github//OpencvAlgorithmPractice//OpencvAlgorithmPractice//images//blackandwhite.png", cv::IMREAD_COLOR);


	while (true) {
		oc::circleFinder finder1;
		oc::circle center;
		center.x = 230;
		center.y = 200;

		cv::Mat copy1 = color1.clone();
		oc::circle result1 = finder1.compute(image1, center, 100, 200, 15, 50, 10);
		finder1.draw(copy1, result1);


	
		oc::lineFinder finder2;

		oc::rect rect;
		rect.x = 589;
		rect.y = 60;
		rect.width = 100;
		rect.height = 700;
		rect.angle = 50;

		cv::Mat copy2 = color2.clone();

		oc::line result2 = finder2.compute(image2, rect, 50, 2, 0.99);
		finder2.draw(copy2, rect);
		finder2.draw(copy2, result2, cv::Scalar(0, 0, 255));



		cv::imshow("original1", image1);
		cv::imshow("color1", copy1);

		cv::imshow("original2", image2);
		cv::imshow("color2", copy2);

		cv::waitKey(0);
	}

}
