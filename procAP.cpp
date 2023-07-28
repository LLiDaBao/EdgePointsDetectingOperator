#include "procAP.hpp"

namespace detAP
{
	void preProcess(const cv::Mat& src, cv::Mat& thresh, int threshold)
	{
		cv::Mat temp;
		if (src.type() == CV_8UC3)
		{
			cv::cvtColor(src, temp, cv::COLOR_BGR2GRAY);
		}
		else
		{
			temp = src;
		}

		if (threshold == -1)
		{
			cv::threshold(temp, thresh, threshold, 255, cv::THRESH_OTSU);
		}

		else
		{
			cv::threshold(temp, thresh, threshold, 255, cv::THRESH_BINARY);
		}

		// morphology operation to delte thin edge
		auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
		cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 3);
	}

	cv::Vec3f approxiLocateAP(const cv::Mat& thresh, float sample_length2, int num_sample)
	{
		std::vector<cv::Point2f> base_points1;	// right-most edge
		std::vector<cv::Point2f> base_points2;	// middle edge
		std::vector<cv::Point2f> base_points3;	// left-most edge

		const float center_row = rand() % thresh.rows;

		for (size_t i = 0; i != num_sample; ++i)
		{
			cv::RotatedRect r_rect;
			edgedet::MeasureHandle m_handle;

			std::vector<cv::Point2f> edges;
			std::vector<double> amplitudes, distances;

			edgedet::genMeasureRectangle2(m_handle, center_row, thresh.cols / 2, 0.0, thresh.cols / 3, sample_length2);

			edges.clear();
			edgedet::measurePos(thresh, m_handle, edges, amplitudes, distances, 80.0, 3, 1.0f, edgedet::Transition::ALL, edgedet::Select::ALL);

			if (edges.size() < 2)
			{
				//std::cerr << "Sample Failed...\nStart Next Sample...\n" << std::endl;
				continue;
			}

			base_points1.emplace_back(edges.back());
			base_points2.emplace_back(edges[edges.size() - 2]);
			base_points3.emplace_back(
				base_points2.back().x - 240,
				base_points1.back().y
			);
		}

		cv::Vec3f base_x = cv::Vec3f(0, 0, 0);

		for (size_t k = 0; k != base_points1.size(); ++k)
		{
			base_x[0] += base_points1[k].x;
			base_x[1] += base_points2[k].x;
			base_x[2] += base_points3[k].x;
		}
		base_x[0] /= base_points1.size();
		base_x[1] /= base_points2.size();
		base_x[2] /= base_points3.size();

		return base_x;
	}

	void detectAP(const cv::Mat& src, std::vector<std::vector<cv::Point2f>>& edgesAP, const cv::Vec3f approx_base, int sample_length2, int num_sample)
	{
		edgesAP.clear();
		edgesAP.resize(3);
		constexpr float sample_length1 = 30;

		const float& right_base = approx_base[0];
		const float& middle_base = approx_base[1];
		const float& left_base = approx_base[2];

		const float& start_row = sample_length2;

		std::vector<cv::Point2f>& right_edge = edgesAP[0];	// result
		std::vector<cv::Point2f>& middle_edge = edgesAP[1];	// result
		std::vector<cv::Point2f>& left_edge = edgesAP[2];	// result

		std::vector<cv::Point2f> edges;
		std::vector<double> amplitudes, distances;

		cv::Mat canvas = src.clone();
		if (src.channels() == 1)
			cv::cvtColor(canvas, canvas, cv::COLOR_GRAY2BGR);

		for (int i = 0; i < num_sample < src.rows; ++i)
		{
			if (start_row + 2 * i * sample_length2 >= src.rows)
				break;

			cv::RotatedRect r_rect1, r_rect2, r_rect3;
			edgedet::MeasureHandle m_handle1, m_handle2, m_handle3;
			edgedet::genMeasureRectangle2(m_handle1, start_row + 2 * i * sample_length2, right_base, 0.0, sample_length1, sample_length2);

			for (int i = 0; i != 4; ++i)
				cv::line(canvas, m_handle1.points[i], m_handle1.points[(i + 1) % 4], cv::Scalar(0, 0, 180));

			edgedet::genMeasureRectangle2(m_handle2, start_row + 2 * i * sample_length2, middle_base, 0.0, sample_length1, sample_length2);

			for (int i = 0; i != 4; ++i)
				cv::line(canvas, m_handle2.points[i], m_handle2.points[(i + 1) % 4], cv::Scalar(0, 0, 180));

			edgedet::genMeasureRectangle2(m_handle3, start_row + 2 * i * sample_length2, left_base, 0.0, sample_length1, sample_length2);

			for (int i = 0; i != 4; ++i)
				cv::line(canvas, m_handle3.points[i], m_handle3.points[(i + 1) % 4], cv::Scalar(0, 0, 180));

			// right-most edge detection
			bool flag1 = edgedet::measurePos(src, m_handle1, edges, amplitudes, distances, 30.0, 3, 1.0f, edgedet::Transition::NEGATIVE, edgedet::Select::LAST);

			if (flag1 && !edges.empty())
			{
				//std::cerr << "#1 Failed..." << std::endl;
				right_edge.emplace_back(edges.back());
			}
			edges.clear();

			// middle edge detection
			bool flag2 = edgedet::measurePos(src, m_handle2, edges, amplitudes, distances, 60.0, 3, 1.0f, edgedet::Transition::POSITIVE, edgedet::Select::LAST);

			if (flag2 && !edges.empty())
			{
				//std::cerr << "#2 Failed..." << std::endl;
				middle_edge.emplace_back(edges.back());
			}
			edges.clear();

			// left edge detection
			bool flag3 = edgedet::measurePos(src, m_handle3, edges, amplitudes, distances, 30.0, 3, 1.0f, edgedet::Transition::POSITIVE, edgedet::Select::FIRST);

			if (flag3 && !edges.empty())
			{
				//std::cerr << "#3 Failed..." << std::endl;
				left_edge.emplace_back(edges.back());
			}
		}
		//for (size_t k = 0; k != right_edge.size(); ++k)
		//	cv::circle(canvas, right_edge[k], 2, cv::Scalar(180, 180, 0), -1);
		//
		//for (size_t k = 0; k != middle_edge.size(); ++k)
		//	cv::circle(canvas, middle_edge[k], 2, cv::Scalar(0, 180, 0), -1);

		//for (size_t k = 0; k != middle_edge.size(); ++k)
		//	cv::circle(canvas, left_edge[k], 2, cv::Scalar(0, 180, 180), -1);

	}

	void fitLinAP(std::vector<std::vector<cv::Point2f>>& edgesAP, std::vector<cv::Vec4f>& lines)
	{
		lines.resize(3);
		if (edgesAP[0].size() > 5)
			cv::fitLine(edgesAP[0], lines[0], cv::DIST_L1, 0, 1e-2, 1e-2);
		if (edgesAP[1].size() > 5)
			cv::fitLine(edgesAP[1], lines[1], cv::DIST_L1, 0, 1e-2, 1e-2);
		if (edgesAP[2].size() > 5)
			cv::fitLine(edgesAP[2], lines[2], cv::DIST_L1, 0, 1e-2, 1e-2);
	}

}	// namespace
