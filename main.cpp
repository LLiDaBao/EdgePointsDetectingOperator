#include "edgedet.h"

using namespace std;
int main(int argc, char** argv) {
	cv::Mat image = cv::imread("./test.jpg");

	cv::RotatedRect r_rect;
	edgedet::MeasureHandle m_handle;
	edgedet::genMeasureRectangle2(r_rect, m_handle, 150, 600, 5.0, 50.0f, 10.0);
	//edgedet::genMeasureArc(m_handle, 150, 600, 50, 30, 60, 30);
	std::vector<cv::Point2f> edges;
	std::vector<float> amplitudes;
	std::vector<float> distances;
	edgedet::measurePos(image, m_handle, edges, amplitudes, distances, 3, 0.1f, 30.0f);
}
