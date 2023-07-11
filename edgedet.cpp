#include "edgedet.h"

using std::cout;
using std::cerr;
using std::endl;

namespace edgedet {

	cv::Point nearestNeighbor(const cv::Point2f& point)
	{
		int floor_x = cvFloor(point.x), floor_y = cvFloor(point.y);
		float diff_x = point.x - floor_x, diff_y = point.y - floor_y;
		cv::Point2f nearest_point;

		if (diff_x <= 0.5 && diff_y <= 0.5)
		{
			nearest_point = cv::Point(floor_x, floor_y);
		}
		else if (diff_x <= 0.5 && diff_y > 0.5)
		{
			nearest_point = cv::Point(floor_x, floor_y + 1);
		}
		else if (diff_x > 0.5 && diff_y <= 0.5)
		{
			nearest_point = cv::Point(floor_x + 1, floor_y);
		}
		else
		{
			nearest_point = cv::Point(floor_x + 1, floor_y + 1);
		}
		return nearest_point;
	}

	cv::Point interpolate(const cv::Point2f& point, Interpolation inter)
	{
		// 给定浮点型Point，找到其最近的点

		cv::Point inter_point;

		switch (inter)
		{
		case Interpolation::NEAREST_NEIGHBOR:
			inter_point = nearestNeighbor(point);

			break;

		case Interpolation::BILINEAR:

			break;

		case Interpolation::BICUBIC:

			break;
		}

		return inter_point;
	}

	void GaussianBlur(const cv::Mat& avg_grays, cv::Mat& blur, int ksize, float sigma)
	{

		/******************************************************************************************
		** Convolve the Avarage Gray Value along the Straight Edge Perpendicular to Profile Line
		* param1: avg_grays: 1 x N Avarage Gray Value Matrix
		* param2: smooth: Output Array of Smoothing Result
		* param3: kernel: Gaussian kernel
		******************************************************************************************/
		cv::Mat kernel = cv::getGaussianKernel(ksize, sigma, CV_32F).t();
		cv::flip(kernel, kernel, -1);	// flip kernel to implement real convolution

		// convolution operation
		cv::filter2D(avg_grays, blur, CV_32F, kernel,
			cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
	}

	void differentiate1st(const cv::Mat& avg_grays, cv::Mat& deriv)
	{
		/*
		* 以下代码等价于
		*	cv::Sobel(avg_grays, deriv, CV_32F, 1, 0, 1, 1.0, 0.0, cv::BORDER_REPLICATE)；
		*/

		// initialize Sobel kernel
		cv::Mat kernel = (cv::Mat_<float>(1, 3) << -1.0f, 0.0f, 1.0f);

		// correlate operation
		cv::filter2D(avg_grays, deriv, CV_32F, kernel,
			cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
		deriv *= (sqrt(2.0 * CV_PI) * 0.5f);
	}

	bool genMeasureArc(MeasureHandle& m_handle, float center_row, float center_col, float radius, float angle_start,
		float angle_extent, float annulus_radius, Interpolation inter)
	{
		// restriction of params
		if (!(-180.0f <= angle_start && angle_start <= 180.0f &&
			-180.0f <= angle_start + angle_extent &&
			angle_start + angle_extent <= 180.0f &&
			annulus_radius <= radius))
		{
			cerr << "Error Input...\n\n" << endl;
			return false;
		}

		m_handle.type = MeasureHandleType::ARC;
		m_handle.center = cv::Point2f(center_col, center_row);
		m_handle.radius = radius;
		m_handle.angle_start = angle_start;
		m_handle.angle_extent = angle_extent;
		m_handle.annulus_radius = annulus_radius;
		m_handle.interpolation = inter;

		return true;
	}

	void RotatedRect2MeasureHandle(const cv::RotatedRect& r_rect, MeasureHandle& m_handle, Interpolation interpolation)
	{
		// type conversion
		m_handle.center = r_rect.center;
		m_handle.type = MeasureHandleType::RECTANGLE;
		m_handle.interpolation = interpolation;
		m_handle.phi = -r_rect.angle;
		m_handle.length1 = r_rect.size.width / 2;
		m_handle.length2 = r_rect.size.height / 2;


		cv::Point2f vertices[4];
		r_rect.points(vertices);	// get 4 corner points of rectangle
		for (int i = 0; i != 4; ++i)
			m_handle.points.emplace_back(vertices[i]);
	}

	void genMeasureRectangle2(cv::RotatedRect& r_rect, MeasureHandle& h_measure,
		int center_row, int center_col, float phi, float length1, float length2, Interpolation interpolation)
	{
		assert(-180.0f <= phi <= 180.0f);
		//**** cv::RotatedRect, x axis positive direction is left to right, 
		//**** y axis positive direction is top to bottom. 
		r_rect = cv::RotatedRect(cv::Point2f(center_col, center_row),
			cv::Size2f(2.0f * length1, 2.0f * length2), -phi);

		RotatedRect2MeasureHandle(r_rect, h_measure, interpolation);
	}

	std::pair<cv::Point2f, cv::Point2f> getProfileLine(const MeasureHandle& m_handle)
	{
		std::pair<cv::Point2f, cv::Point2f> profile_line;
		profile_line.first = cv::Point2f(
			(m_handle.points[0].x + m_handle.points[1].x) * 0.5f,
			(m_handle.points[0].y + m_handle.points[1].y) * 0.5f
		);
		profile_line.second = cv::Point2f(
			(m_handle.points[2].x + m_handle.points[3].x) * 0.5f,
			(m_handle.points[2].y + m_handle.points[3].y) * 0.5f
		);

		if (profile_line.first.x == profile_line.second.x
			&& profile_line.first.y > profile_line.second.y
			|| profile_line.first.x > profile_line.second.x)
		{
			std::swap(profile_line.first, profile_line.second);
		}

		return profile_line;
	}

	cv::Point2f getCurrPoint(float _k, const cv::Point2f& through_point, float _x)
	{
		// y - y0 = k * (x - x0)
		return cv::Point2f(
			_x, _k * (_x - through_point.x) + through_point.y
		);
	}

	bool perpendAvgGrays(const cv::Mat& src, cv::Mat& avg_grays, std::vector<cv::Point2f>& center_pts, const MeasureHandle& m_handle, bool display)
	{
		/**** Source Image Checking ****/
		if (src.empty() || src.type() != CV_8UC1 && src.type() != CV_8UC3)
		{
			cerr << "Error...\nCheck Source Image...\n\n" << endl;
			return false;
		}

		/**** Convert to Gray ****/
		cv::Mat gray;
		if (src.channels() == 1)
		{
			src.convertTo(gray, CV_32F);
		}
		else
		{
			cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
			gray.convertTo(gray, CV_32F);
		}

		/***************************************/
		/**** Calculate Avarage Pixel Value ****/
		/***************************************/

		const float step = 1.0f;
		cv::Mat canvas = src.clone();	// image show

		center_pts.clear();	// clear useless data

		if (m_handle.type == MeasureHandleType::ARC)	// arc type
		{
			// counterclockwise direction
			float arc_start_angle = m_handle.angle_start;
			float arc_end_angle = m_handle.angle_start + m_handle.angle_extent;
			if (arc_start_angle > arc_end_angle)
			{
				std::swap(arc_start_angle, arc_end_angle);
			}

			const float delta_angle = 1.0f;

			avg_grays = cv::Mat::zeros(cv::Size(cvRound((arc_end_angle - arc_start_angle) / delta_angle), 1), CV_32F);

#pragma omp parallel for
			for (int i = 0; i != cvRound((arc_end_angle - arc_start_angle) / delta_angle); ++i)
			{
				// calculate avarge gray value along the current perpendicular line
				float avg = 0.0f;
				float arc_curr_angle = arc_start_angle + i * delta_angle;

				for (int j = 0; j != static_cast<int>((m_handle.radius - m_handle.annulus_radius) / step); ++j)
				{
					cv::Point2f curr_perpend_pt = cv::Point2f(
						m_handle.center.x + (m_handle.annulus_radius + j * step) * cos(-CV_PI / 180.0 * arc_curr_angle),
						m_handle.center.y + (m_handle.annulus_radius + j * step) * sin(-CV_PI / 180.0 * arc_curr_angle)
					);

					const auto& inter_point = interpolate(curr_perpend_pt);	// interpolation
					avg += gray.at<float>(inter_point.y, inter_point.x);

					if (display)
					{
						cv::circle(canvas, curr_perpend_pt, 1, cv::Scalar(0, 100, 100), -1);
					}
				}

				avg_grays.at<float>(0, i) = (avg / (m_handle.radius - m_handle.annulus_radius) / step);
				center_pts.emplace_back(
					m_handle.center.x + 0.5 * (m_handle.annulus_radius + m_handle.radius) * cos(-CV_PI / 180.0 * arc_curr_angle),
					m_handle.center.y + 0.5 * (m_handle.annulus_radius + m_handle.radius) * sin(-CV_PI / 180.0 * arc_curr_angle)
				);
			}

			if (display)
			{
				for (float arc_curr_angle = arc_start_angle;
					arc_curr_angle <= arc_end_angle; arc_curr_angle += delta_angle)
				{
					const cv::Point2f& curr_pt =
						cv::Point2f(
							m_handle.center.x + m_handle.radius * cos(-CV_PI / 180.0 * arc_curr_angle),
							m_handle.center.y + m_handle.radius * sin(-CV_PI / 180.0 * arc_curr_angle));

					if (arc_curr_angle == arc_start_angle)
					{
						cv::line(canvas, m_handle.center, curr_pt, cv::Scalar(0, 0, 0), 1);
						cv::circle(canvas, m_handle.center, 3, cv::Scalar(0, 0, 255), -1);
					}
					cv::circle(canvas, curr_pt, 1, cv::Scalar(0, 0, 0), -1);
				}
				cv::imshow("Canvas", canvas);
				cv::waitKey(1000);
			}
		}

		else if (m_handle.type == MeasureHandleType::RECTANGLE)
		{
			// limit rotated rectangle phi in range [-90.0, 90.0]
			float limit_phi = m_handle.phi;
			if (m_handle.phi < -90.0f)
			{
				limit_phi += 180.0f;
			}
			else if (m_handle.phi > 90.0f)
			{
				limit_phi -= 180.0f;
			}

			avg_grays = cv::Mat::zeros(cv::Size(cvRound(2.0 * m_handle.length1), 1), CV_32F);

			auto profile_line = getProfileLine(m_handle);

			float slope = 0.0f, vertical_slope = 0.0f;
			if (abs(limit_phi) != 90.0f)	// if not up-left rectangle, initialize
			{
				slope = tan(-CV_PI / 180.0f * limit_phi);

				if (limit_phi != 0.0f)
				{
					vertical_slope = (limit_phi < 0)
						? tan(-CV_PI / 180.0f * (limit_phi + 90.0f))
						: tan(-CV_PI / 180.0f * (limit_phi - 90.0f));
				}
			}

			// parallel
#pragma omp parallel for
			for (int i = 0; i != cvRound(2.0 * m_handle.length1); ++i)
			{
				cv::Point2f curr_profile_pt;
				// initialize current profile point
				if (abs(limit_phi) != 90.0f)
				{
					curr_profile_pt = getCurrPoint(
						slope, profile_line.first,
						profile_line.first.x + cos(-CV_PI / 180.0f * limit_phi) * i
					);
				}
				else
				{
					curr_profile_pt =
						cv::Point2f(profile_line.first.x, profile_line.first.y + i);
				}

				float avg = 0.0f;
				int skip_num = 0;

				for (int j = 0; j != cvRound(2.0 * m_handle.length2 / step); ++j)
				{
					float start_perpend_x = -1.0f;
					cv::Point2f curr_perpend_pt;

					// initialize points of perpendicular line
					if (abs(limit_phi) != 90.0f && limit_phi != 0.0f)
					{
						start_perpend_x =
							curr_profile_pt.x - 1.0f * m_handle.length2
							* sin(CV_PI / 180.0f * abs(limit_phi));

						curr_perpend_pt = getCurrPoint(
							vertical_slope, curr_profile_pt,
							start_perpend_x + sin(CV_PI / 180.0f * abs(limit_phi)) * j * step);
					}
					else if (limit_phi == 0.0f)
					{
						start_perpend_x = curr_profile_pt.x;

						curr_perpend_pt =
							cv::Point2f(start_perpend_x, curr_profile_pt.y - m_handle.length2 + j * step);
					}
					else
					{
						start_perpend_x =
							curr_profile_pt.x - m_handle.length2;

						curr_perpend_pt = cv::Point2f(start_perpend_x + j * step, curr_profile_pt.y);
					}

					const auto& inter_point = interpolate(curr_perpend_pt);

					// check if is legal point
					if (inter_point.x < 0 || inter_point.y < 0)
					{
						cerr << "\nIllegal Postion...\nSkip Current Point...\n\n" << endl;
						++skip_num;
						continue;
					}

					avg += gray.at<float>(inter_point.y, inter_point.x);

					if (display)	// display perpendicular line
					{
						if (j == 0)
							for (int idx = 0; idx != 4; ++idx)
								cv::line(canvas, m_handle.points[idx],
									m_handle.points[(idx + 1) % 4], cv::Scalar(0, 0, 255), 2);

						cv::circle(canvas, inter_point, 1, cv::Scalar(255, 0, 0), -1);

						cv::imshow("Canvas", canvas);
						cv::waitKey(1);
					}
				}

				avg_grays.at<float>(0, i) = static_cast<float>(avg / (m_handle.length2 * 2 / step - skip_num));
				center_pts.emplace_back(curr_profile_pt);
			}
		}

		cv::destroyAllWindows();
		return true;
	}

	double edgeSubPixel(const cv::Mat& deriv, double& offset, int local_max_ind)
	{
		// Use Devernay Method to Calculate Subpixel Position
		if (local_max_ind == 0 || local_max_ind == deriv.cols - 1)	// neet at least 3 points
		{
			offset = 0.0;
			return deriv.at<float>(0, local_max_ind);
		}

		cv::Mat coeffs;	// c0, c1, c2
		cv::Mat X = (cv::Mat_<float>(3, 3) << 
			1.0f, 1.0f * (local_max_ind - 1), 1.0f * (local_max_ind - 1) * (local_max_ind - 1),
			1.0f, 1.0f * local_max_ind, 1.0f * local_max_ind * local_max_ind,
			1.0f, 1.0f * (local_max_ind + 1), 1.0f * (local_max_ind + 1) * (local_max_ind + 1)
		);
		cv::Mat Y = (cv::Mat_<float>(3, 1) << 
			deriv.at<float>(0, local_max_ind - 1),
			deriv.at<float>(0, local_max_ind),
			deriv.at<float>(0, local_max_ind + 1)
		);

		coeffs = X.inv() * Y;

		offset = -0.5 * coeffs.at<float>(1, 0) / coeffs.at<float>(2, 0);
		return (4.0 * coeffs.at<float>(2, 0) * coeffs.at<float>(0, 0) - 
			coeffs.at<float>(1, 0) * coeffs.at<float>(1, 0)) / (4.0 * coeffs.at<float>(2, 0));
	}

	bool measurePos(const cv::Mat& src, const MeasureHandle& m_handle, 
				std::vector<cv::Point2f>& edges, std::vector<float>& amplitudes, 
				std::vector<float>& distances, int ksize, float sigma, float threshold,
				Transition trans, Select select, Interpolation inter, bool display)
	{	
		/**** Calculate Avarage Pixel Value ****/
		cv::Mat avg_grays;
		std::vector<cv::Point2f> center_pts;
		perpendAvgGrays(src, avg_grays, center_pts, m_handle, display);
		
		/**** Gaussian Blur and Calculate Derivation ****/
		std::cout << "\nGaussian Smoothing and Differentiate...\n" << std::endl;
		GaussianBlur(avg_grays, avg_grays, ksize, sigma);
		cv::Mat deriv = avg_grays;
		differentiate1st(avg_grays, deriv);
		std::cout << "Done...\nDerivation is:\n" 
			<< deriv << "\n\n" << std::endl;

		/**** Filter Points ****/
		std::cout << "Detecting Edge...\n" << std::endl;
		std::vector<int> candidates;
		switch (trans)
		{
		case Transition::ALL:
			for (int col = 0; col != deriv.cols; ++col)
			{
				if (abs(deriv.at<float>(0, col)) > threshold)
				{
					candidates.push_back(col);
				}
			}

			break;

		case Transition::POSITIVE:
			for (int col = 0; col != deriv.cols; ++col)
			{
				if (deriv.at<float>(0, col) < 0.0 &&
					abs(deriv.at<float>(0, col)) > threshold)
				{
					candidates.push_back(col);
				}
			}

			break;

		case Transition::NEGATIVE:
			for (int col = 0; col != deriv.cols; ++col)
			{
				if (deriv.at<float>(0, col) >= 0.0 &&
					abs(deriv.at<float>(0, col)) > threshold)
				{
					candidates.push_back(col);
				}
			}

			break;
		}
		
		if (candidates.empty())
		{
			std::cerr << "\nNo Edge Points Found...\n\n" << std::endl;
			return false;
		}

		const int MAX_CONSECUTIVE_NUM = 4;
		if (candidates.size() >= 2)
		{
			std::vector<std::vector<int>> partitions;
			std::vector<int> part = { candidates.front() };

			auto profile_line = getProfileLine(m_handle);

			// split source vector to some partitions
			for (int i = 1; i != candidates.size(); ++i)
			{
				const auto& pre_col = candidates[i - 1];
				const auto& curr_col = candidates[i];
				const float& pre_deriv = deriv.at<float>(0, pre_col);
				const float& curr_deriv = deriv.at<float>(0, curr_col);

				if (//part.size() != MAX_CONSECUTIVE_NUM &&
					curr_col - pre_col <= MAX_CONSECUTIVE_NUM &&
					(pre_col < 0 && curr_col < 0 ||
						pre_col >= 0 && curr_col >= 0))
				{
					part.push_back(curr_col);
				}
				else
				{
					partitions.push_back(part);
					part.clear();
					part.push_back(curr_col);
				}
			}
			if (!part.empty())
			{
				partitions.push_back(part);
			}

			std::cout << "\nThe Number of Partitions is :" 
				<< partitions.size() << std::endl;

			// calculate local maximum of each partition
			for (const auto& partition : partitions)
			{
				int local_max_ind = 0;
				float local_max = 0.0f;
				for (size_t k = 0; k != partition.size(); ++k)
				{
					const auto& curr_d = deriv.at<float>(0, partition[k]);
					if (local_max <= abs(curr_d))
					{
						local_max_ind = partition[k];
						local_max = abs(curr_d);
					}
				}

				cv::Point2f edge_pt;
				double offset = 0.0;
				double amp = edgeSubPixel(deriv, offset, local_max_ind);

				if (profile_line.first.x != profile_line.second.x)
				{
					float k = (profile_line.first.y - profile_line.second.y) /
						(profile_line.first.x - profile_line.second.x);
					edge_pt = getCurrPoint(k, profile_line.first, profile_line.first.x + offset);
				}
				else
				{
					edge_pt = cv::Point2f(
						profile_line.first.x,
						profile_line.first.y + offset
					);
				}

				edges.emplace_back(edge_pt);
				//edges.emplace_back(center_pts[local_max_ind]);
				amplitudes.push_back(amp);
			}
		}
		else if (candidates.size() == 1)
		{
			edges.emplace_back(center_pts[candidates.front()]);
			amplitudes.push_back(deriv.at<float>(0, candidates.front()));
		}

		cv::Point2f temp_pt;
		float temp_amp = 0.0;
		switch (select)
		{
		case Select::ALL:
			break;

		case Select::FIRST:
			temp_pt = edges.front(), temp_amp = amplitudes.front();
			edges.clear(), amplitudes.clear();
			edges.emplace_back(temp_pt), amplitudes.push_back(temp_amp);
			break;

		case Select::LAST:
			temp_pt = edges.back(), temp_amp = amplitudes.back();
			edges.clear(), amplitudes.clear();
			edges.emplace_back(temp_pt), amplitudes.push_back(temp_amp);
			break;
		}

		if (edges.size() >= 2)
		{
			for (size_t k = 1; k != edges.size(); ++k)
			{
				distances.push_back(
					sqrt(
						pow(edges[k].x - edges[k - 1].x, 2.0) +
						pow(edges[k].y - edges[k - 1].y, 2.0))
				);
			}
		}


		// print result

		for (size_t i = 0; i != edges.size(); ++i)
		{
			if (i == 0)
			{
				std::cout << "\nEdge Point is:\n";
				std::cout << edges[i];
				continue;
			}
			std::cout << ", " << edges[i] ;
		}
		
		for (size_t i = 0; i != amplitudes.size(); ++i)
		{
			if (i == 0)
			{
				std::cout << "\nAmplitude is:\n";
				std::cout << amplitudes[i];
				continue;
			}
			std::cout << ", " << amplitudes[i];
		}

		if (!distances.empty())
		{
			for (size_t i = 0; i != distances.size(); ++i)
			{
				if (i == 0)
				{
					std::cout << "\nDistance is:\n";
					std::cout << distances[i];
					continue;
				}
				std::cout << ", " << distances[i];
			}
			std::cout << std::endl;
		}

		bool display_result = true;
		if (display_result)
		{
			cv::Mat canvas = src.clone();
			for (const auto& pt : edges)
			{
				int b = rand() % 256,
					g = rand() % 256,
					r = rand() % 256;
				cv::circle(canvas, pt, 3, cv::Scalar(b, g, r), -1);
			}
			cv::imshow("Result Points", canvas);
			cv::waitKey(0);
		}

		return true;
	}


}	// namespace edgedet