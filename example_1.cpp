#include "edgedet.hpp"
#include <string>


// Example1: Straight Line Edge Detecting

int main(int argc, char** argv)
{
	cv::Mat src = cv::imread("./images/color_barcode.jpg", 0);

	cv::Mat canvas = src.clone();
	cv::cvtColor(canvas, canvas, cv::COLOR_GRAY2BGR);
	cv::RotatedRect r_rect;

	// Set your own value
	int center_row = 50, center_col = 950;
	double phi = 0.0;
	float lenght1 = 60, length2 = 10;
	int find_num = 30;
	
	cv::Mat result = cv::Mat::zeros(cv::Size(4, find_num), CV_32F); // save result 

	std::vector<cv::Point2f> edge_points;
	std::vector<double> amplitudes;
	std::vector<double> distances;
	for (int i = 0; i != find_num; ++i)
	{

		edgedet::MeasureHandle m_handle;
		edgedet::genMeasureRectangle2(r_rect, m_handle, center_row + 2 * length2 * i, center_col, phi, lenght1, length2);
		edgedet::measurePos(src, m_handle, edge_points,amplitudes, distances,
			30.0, 3, 1.0, edgedet::Transition::POSITIVE, edgedet::Select::FIRST);

		// draw
		for (const auto& pt : edge_points)
		{
			int b = rand() % 256, g = rand() % 256, r = rand() % 256;
			cv::circle(canvas, pt, 6, cv::Scalar(b, g, r), -1);
		};
	}

	cv::namedWindow("Edge Points", cv::WINDOW_FREERATIO);
	cv::imshow("Edge Points", canvas);
	cv::waitKey(0);
	return 0;
}
