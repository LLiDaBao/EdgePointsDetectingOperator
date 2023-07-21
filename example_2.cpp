#include "edgedet.hpp"
#include <string>

//Example: 

int main(int argc, char** argv)
{
	cv::Mat src = cv::imread("./images/loop.jpg", 0), dst;
	cv::Mat map_x, map_y;
	std::vector<cv::Point2f> points;

	double timer = cv::getTickCount();
	edgedet::ellipseEdgeDetect(src,  cv::Point2f(src.cols / 2, src.rows / 2), dst, points);
	std::cout << "Detecting Ellipse Edge Points Consume: " << (cv::getTickCount() - timer) * 1000.0 / cv::getTickFrequency() << " ms\n" << std::endl;
	cv::Mat canvas;
	cv::cvtColor(src, canvas, cv::COLOR_GRAY2BGR);
	for (const auto& pt : points)
		cv::circle(canvas, pt, 1, cv::Scalar(0, 0, 255), -1);
	cv::imshow("canvas", canvas);
	cv::waitKey(0);
	return 0;
}
