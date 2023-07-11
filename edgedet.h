#pragma once
#include <opencv.hpp>
#include <utility>
#include <imgproc/imgproc.hpp>
#include <omp.h>

namespace edgedet {

	// enum type declaration
	enum class Interpolation {
		NEAREST_NEIGHBOR,
		BILINEAR,
		BICUBIC
	};

	enum class Transition {
		ALL,
		NEGATIVE,
		POSITIVE
	};

	enum class Select {
		ALL,
		FIRST,
		LAST
	};

	enum class InterplationC
	{
		LAGRANGE,
		NEWTON,
		HERMITE
	};

	enum class MeasureHandleType
	{
		ARC,
		RECTANGLE
	};

	/******1D ��Ե��ȡ����******/
	/*
	* 1. ��gen_mesure_rectangle2����MesureHandle���
	* 2. �ش�ֱ��������(profile line)�������ƽ���Ҷȣ�������������Ϊ 1 pixel���õ�һά��ɢ���飬���и�˹�˲�
	* 3. ��2. �����һ�׵���
	* 4. ��ȡ��Ե
	*/

	class MeasureHandle 
	{
	public:	//base part data members
		cv::Point2i center;	// the center of rotate rectangle
		MeasureHandleType type;
		Interpolation interpolation;

	public:	// rect part data members
		double phi;	// rad
		float length1;	// half of longer axis length 
		float length2;	
		std::vector<cv::Point2f> points;
		
	public:	// arc part data members
		float radius;
		float angle_start;
		float angle_extent;
		float annulus_radius;
	};

	extern cv::Point nearestNeighbor(const cv::Point2f& point);
	extern cv::Point interpolate(const cv::Point2f& point, Interpolation inter = Interpolation::NEAREST_NEIGHBOR);
	extern void GaussianBlur(const cv::Mat& avg_grays, cv::Mat& blur, int ksize, float sigma);
	extern void differentiate1st(const cv::Mat& avg_grays, cv::Mat& deriv);

	extern bool genMeasureArc(MeasureHandle& m_handle, float center_row, float center_col, float radius, float angle_start,
		float angle_extent, float annulus_radius, Interpolation inter = Interpolation::NEAREST_NEIGHBOR);

	extern void RotatedRect2MeasureHandle(const cv::RotatedRect& r_rect, MeasureHandle& h_measure, Interpolation interpolation);
	
	extern void genMeasureRectangle2(cv::RotatedRect& r_rect, MeasureHandle& m_handle, int center_row, int center_col,
		float phi, float length1, float length2, Interpolation interpolation = Interpolation::NEAREST_NEIGHBOR);

	extern std::pair<cv::Point2f, cv::Point2f> getProfileLine(const MeasureHandle& m_handle);

	extern cv::Point2f getCurrPoint(float _k, const cv::Point2f& through_point, float _x);

	
	extern bool perpendAvgGrays(const cv::Mat& gray, cv::Mat& avg_grays, std::vector<cv::Point2f>& center_pts, const MeasureHandle& m_handle, bool display = true);
	extern double edgeSubPixel(const cv::Mat& deriv, double& offset, int local_max_ind);
	extern bool measurePos(const cv::Mat& src, const MeasureHandle& m_handle, 
						std::vector<cv::Point2f>& edges, std::vector<float>& amplitudes, 
						std::vector<float>& distances, int ksize = 3, float sigma = 1.0, float threshold = 30.0f, 
						Transition trans = Transition::ALL, Select select = Select::ALL, Interpolation inter = Interpolation::NEAREST_NEIGHBOR, 
						bool display = true);
}