#pragma once

#include "edgedet.hpp"
#include <imgproc/imgproc.hpp>
#include <functional>
#include <vector>

namespace detAP	
{
	/*******************************************************************/
	/****************** Welding Components Detecting *******************
 	/************* to extract 3 edges of welding components ************
	********************************************************************


	/* @brief OTSU threshold and open morphology operation to extract ROI.
	* 
	* @param src: 8-bit input image.
	* @param thresh: binary output image.
	* @param threshold: threshold value, if is -1, use OTSU method, else threshold by given threshold.
	*/
	extern void preProcess(const cv::Mat& src, cv::Mat& thresh, int threshold = -1);

	/* @brief Locate approximate X-position of AP's edges.
	* 
	* @param thresh: input binary image.
	* @param sample_length2: parameter length2 of MeasureHandle class.
	* @param num_sample: the number of points of random samples.
	* 
	* @reutrn 3 X-positions of AP's edges.
	*/
	extern cv::Vec3f approxiLocateAP(const cv::Mat& thresh, float sample_length2, int num_sample = 10);

	/* @brief Detect AP's 3 edges and get some points of them.
	* 
	* @param src: 8-bit input image.
	* @param edgesAP: output array of edges points of AP.
	* @param approx_base: approximate position of 3 edges of AP.
	* @param sample_length2: length2 parameter of MeasureHandle.
	* @param num_sample: the number of samples along the AP edge.
	*/
	extern void detectAP(const cv::Mat& src, std::vector<std::vector<cv::Point2f>>& edgesAP, const cv::Vec3f approx_base, int sample_length2, int num_sample = 15);

	/* @brief Fit 3 edges by input points.
	*/
	extern void fitLinAP(std::vector<std::vector<cv::Point2f>>& edgesAP, std::vector<cv::Vec4f>& lines);

}	// namespace





void drawResult(const cv::Mat& src, cv::Mat& canvas, float center_row, float center_col, float threshold, 
	float phi, float length1, float length2, int src_rows, edgedet::Transition trans, edgedet::Select select, const cv::Scalar& scalar = cv::Scalar(0, 0, 255))
{

	cv::RotatedRect r_rect;
	edgedet::MeasureHandle m_handle;

	edgedet::genMeasureRectangle2(m_handle, center_row, center_col, phi, length1, length2);

	//for (int i = 0; i != 4; ++i)
	//{
	//	cv::line(canvas, m_handle.points[i], m_handle.points[(i + 1) % 4], cv::Scalar(0, 180, 180));
	//}

	std::vector<cv::Point2f> edge_points;
	std::vector<double> amplitudes;
	std::vector<double> distances;

	edgedet::measurePos(src, m_handle, edge_points, amplitudes, distances, threshold, 3, 1, trans, select);

	for (const auto& pt : edge_points)
	{
		cv::circle(canvas, pt, 3, scalar, -1);
	}
}
