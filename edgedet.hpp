#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>
#include <functional>
#include <iostream>
#include <math.h>
#include <omp.h>

namespace edgedet
{
	// enum declaration

	enum class Interpolation
	{
		NEAREST_NEIGHBOR,
		BILINEAR,
		BICUBIC
	};

	enum class Transition
	{
		ALL,
		NEGATIVE,
		POSITIVE
	};

	enum class Select
	{
		ALL,
		FIRST,
		LAST
	};

	enum class MeasureHandleType
	{
		RECTANGLE,
		ARC
	};

	// MeasureHandle struct definition, from cv::RotatedRect
	class MeasureHandle	// improve in the future
	{
	public:
		cv::Point2f center;
		MeasureHandleType type;
		Interpolation inter;

	public:	// rectangle part
		float phi = 0.0;	// degree
		float length1 = 0.0;
		float length2 = 0.0;
		std::vector<cv::Point2f> points;

	public:	// arc part
		float radius = 0.0f;
		float angle_start = 0.0f;
		float angle_extent = 0.0f;
		float annulus_radius = 0.0f;
	};

	/*******************************************************************/
	/******************** 1D Straight Edge Measure *********************
	********************************************************************
	* [1]: Generate Rotated Rectangle
	* [2]: Calculate Avarage Gray Value along perpendicular to profile line
	* [3]: Gaussian Blur and Calculate 1st Derivation
	* [4]: Threshold and Select
	*/

	/** @brief Points interpolation method.
	*
	* @param point2f: input point.
	* @param inter: interpolation method(e.g. Interpolation::NEARSET_NEIGHTBOR etc.).
	*
	* @return interpolated point.
	*/
	extern cv::Point interpolatePoint(const cv::Point2f& point2f, Interpolation inter = Interpolation::NEAREST_NEIGHBOR);
	extern cv::Point nearestNeighbor(const cv::Point2f& point2f);
	extern cv::Point bilinear(const cv::Point2f& point2f);	// implement in the future
	extern cv::Point bicubic(const cv::Point2f& point2f);	// implement in the future


	/** @brief Get the startand end points of profile line of given rotated rectangle.
	*
	* @param m_handle: MeasureHandle class, see it's definition.
	*
	* @return std::pair type of start and end points of the profile line.
	*/
	extern std::pair<cv::Point2f, cv::Point2f> getProfileLine(const MeasureHandle& m_handle);


	/** @brief Get point by equation y - y0 = k * (x - x0).
	*
	* @param through_pt: point(cv::Point2f) that the line pass through.
	* @param _k: the slope of the line.
	* @param _x: input x.
	*
	* @return current point of input x.
	*/
	extern cv::Point2f getCurrPoint(const cv::Point2f through_pt, float _k, float _x);	// get points in perpendicular line

	// @brief smooth avarage gray amplitude curve.
	extern void GaussianBlur(const cv::Mat& avg_grays, cv::Mat& blur, int ksize, float sigma);

	// @brief calculate 1st derivation.
	extern void differentiate1st(const cv::Mat& avg_grays, float sigma, cv::Mat& deriv);

	// Type conversion.
	extern void RotatedRect2MeasureHandle(const cv::RotatedRect& r_rect, MeasureHandle& h_measure, Interpolation inter);

	// Create MeasureHandle.
	extern bool genMeasureArc(MeasureHandle& m_handle, float center_row, float center_col, float radius, float angle_start,
		float angle_extent, float annulus_radius, Interpolation inter = Interpolation::NEAREST_NEIGHBOR);

	// Create MeasureHandle.
	extern bool genMeasureRectangle2(cv::RotatedRect& r_rect, MeasureHandle& m_handle, float center_row, float center_col,
		float phi, float length1, float length2, Interpolation inter = Interpolation::NEAREST_NEIGHBOR);

	/** @brief Calculate avarage gray amplitudes along the perpendicular line.
	*
	* @param gray: gray image(#CV_8UC1).
	* @param avg_grays: output avarage gray amplitude matrix, type is CV_32F, size is 1xN .
	* @param center_pts: the points of proflie line.
	* @param m_handle: measure handle.
	*
	* @return true means success, else failed.
	*/
	extern bool perpendAvgGrays(const cv::Mat& gray, cv::Mat& avg_grays, std::vector<cv::Point2f>& center_pts, const MeasureHandle& m_handle);

	/** @brief Use Devernay Method to Calculate Subpixel Position, it needs 3 points at least.
	*
	* @param deriv: derivation.
	* @param offset: output of subpixel offset relative to start point of profile line.
	* @param local_max_ind: input of local maximum index.
	*
	* @return local maximum(minimum) of amplitudes.
	*/
	extern double edgeSubPixel(const cv::Mat& deriv, double& offset, int local_max_ind);

	/** @brief Extract edge subpixel postion from region(rectangle shape or annulus shape).
	*
	* @param src: 8-bit input image.
	* @param m_handle: measure handle.
	* @param edge_points: output edge subpixel points.
	* @param amplitudes: output edge points' amplitudes(derivation intensity).
	* @param distances: distances of two neighbor points.
	* @param threshold: amplitude threshold to filter points.
	* @param ksize: Gaussian kernel size.
	* @param sigam: standard deviation of Gaussian kernel.
	* @param transition: transition mode.
	* @param select: select mode.
	* @param inter: interpolation mode.
	*
	* @return true means success, else fail.
	*/
	extern bool measurePos(const cv::Mat& src, const MeasureHandle& m_handle, std::vector<cv::Point2f>& edge_points,
		std::vector<double>& amplitudes, std::vector<double>& distances, float threshold = 30.0, int ksize = 3, float sigma = 1.0f,
		Transition transition = Transition::POSITIVE, Select select = Select::ALL, Interpolation inter = Interpolation::NEAREST_NEIGHBOR);


	/*******************************************************************/
	/************************ Ellipse Detect ***************************
	********************************************************************/

	/** @brief Create arithmetic sequence from start to end.
	*
	* @param start: start value.
	* @param end: end value.
	* @param num_steps: number of the sequence.
	* @param dst: output Mat, size is 1 x num_steps.
	*/
	extern void linspace(const float start, const float end, const int num_steps, cv::Mat& dst);

	// @brief Like numpy works.
	extern void meshgrid(cv::Mat& X, cv::Mat& Y);

	// @breif Get max value index.
	template <typename T = float>
	static int argmax(T* data, int len)
	{
		int max_index = 0;
		T max_val = data[0];
		for (int i = 0; i < len; i++)
		{
			if (max_val < data[i])
			{
				max_index = i;
				max_val = data[i];
			}
		}
		return max_index;
	}

	/** @brief Flatten input image to make ellipse or circle straight. OutputArray dst is src after flatten,
	* and output ellipse's edge points found in image.
	*
	* @param src: 8-bit input image (#CV_8UC1 or #CV_8UC3).
	* @param flatten_center: the center point of input image to flatten.
	* @param dst: output image(#CV_8UC1).
	* @param ellip_points: output. ellipse points found from flatten image.
	*/
	extern void ellipseEdgeDetect(const cv::Mat& src, const cv::Point2f& flatten_center, cv::Mat& dst, std::vector<cv::Point2f>& ellip_points);
}
