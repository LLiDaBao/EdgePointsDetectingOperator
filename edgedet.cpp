#include "edgedet.hpp"

using std::cout;
using std::cerr;
using std::endl;

static double timer = 0.0;

namespace edgedet
{
	cv::Point nearestNeighbor(const cv::Point2f& point2f)
	{
		int floor_x = cvFloor(point2f.x), floor_y = cvFloor(point2f.y);
		float diff_x = point2f.x - floor_x, diff_y = point2f.y - floor_y;

		cv::Point nearest;

		if (diff_x <= 0.5 && diff_y <= 0.5)
		{
			nearest = cv::Point(floor_x, floor_y);
		}
		else if (diff_x <= 0.5 && diff_y > 0.5)
		{
			nearest = cv::Point(floor_x, floor_y + 1);
		}
		else if (diff_x > 0.5 && diff_y <= 0.5)
		{
			nearest = cv::Point(floor_x + 1, floor_y);
		}
		else
		{
			nearest = cv::Point(floor_x + 1, floor_y + 1);
		}

		return nearest;
	}

	cv::Point interpolatePoint(const cv::Point2f& point2f, Interpolation inter)
	{
		cv::Point inter_point;	// return

		switch (inter)
		{
		case Interpolation::NEAREST_NEIGHBOR:
			inter_point = nearestNeighbor(point2f);
			break;

		case Interpolation::BILINEAR:

			break;

		case Interpolation::BICUBIC:

			break;
		}

		return inter_point;
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

		if ((profile_line.first.x == profile_line.second.x &&
			profile_line.first.y > profile_line.second.y) ||
			profile_line.first.x > profile_line.second.x)
		{
			std::swap(profile_line.first, profile_line.second);
		}

		return profile_line;
	}

	cv::Point2f getCurrPoint(const cv::Point2f through_pt, float _k, float _x)
	{
		return cv::Point2f(
			_x, _k * (_x - through_pt.x) + through_pt.y
		);
	}

	void GaussianBlur(const cv::Mat& avg_grays, cv::Mat& blur, int ksize, float sigma)
	{
		cv::Mat kernel =
			cv::getGaussianKernel(ksize, sigma, CV_32F).t();
		cv::flip(kernel, kernel, -1);

		cv::filter2D(avg_grays, blur, CV_32F, kernel,
			cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
		//cv::GaussianBlur(avg_grays, blur, cv::Size(ksize, 1), sigma, 0.0, cv::BORDER_REPLICATE);
	}

	void differentiate1st(const cv::Mat& avg_grays, float sigma, cv::Mat& deriv)
	{
		cv::Mat kernel = (cv::Mat_<float>(1, 3) << -1.0f, 0.0f, 1.0f);

		//cv::flip(kernel, kernel, -1);
		cv::filter2D(avg_grays, deriv, CV_32F, kernel,
			cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
		deriv *= (0.5 * sigma * sqrt(2.0 * CV_PI));	// scale the derivation
	}

	void RotatedRect2MeasureHandle(const cv::RotatedRect& r_rect,
		MeasureHandle& m_handle, Interpolation inter)
	{
		m_handle.type = MeasureHandleType::RECTANGLE;
		m_handle.center = r_rect.center;
		m_handle.phi = -r_rect.angle;
		m_handle.length1 = r_rect.size.width / 2;
		m_handle.length2 = r_rect.size.height / 2;

		cv::Point2f vertices[4];
		r_rect.points(vertices);
		for (int i = 0; i != 4; ++i)
			m_handle.points.emplace_back(vertices[i]);
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
		m_handle.inter = inter;

		return true;
	}

	bool genMeasureRectangle2(cv::RotatedRect& r_rect, MeasureHandle& m_handle,
		float center_row, float center_col, float phi, float length1, float length2, Interpolation inter)
	{
		// restriction of params
		if (!(-180.0f <= phi && phi <= 180.0f))
		{
			cerr << "Error Input...\n\n" << endl;
			return false;
		}

		r_rect = cv::RotatedRect(cv::Point2f(center_col, center_row),
			cv::Size2f(2.0f * length1, 2.0f * length2), -phi);

		RotatedRect2MeasureHandle(r_rect, m_handle, inter);
		return true;
	}

	bool perpendAvgGrays(const cv::Mat& src, cv::Mat& avg_grays, std::vector<cv::Point2f>& center_pts, const MeasureHandle& m_handle)
	{
		/**** Source Image Checking ****/
		if (src.empty() || src.type() != CV_8UC1 && src.type() != CV_8UC3)
		{
			cerr << "Error...\nCheck Source Image...\n\n" << endl;
			return false;
		}

		/**** Convert to Gray ****/

		cv::Mat gray;
		if (src.channels() == 1)
		{
			gray = src;
		}
		else
		{
			cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
		}

		/***************************************/
		/**** Calculate Avarage Pixel Value ****/
		/***************************************/

		const float step = 1.0f;
		center_pts.clear();	// clear useless data

		timer = cv::getTickCount();

		if (m_handle.type == MeasureHandleType::ARC)	// arc type
		{
			// counterclockwise direction

			float arc_start_angle = m_handle.angle_start;
			float arc_end_angle = m_handle.angle_start + m_handle.angle_extent;
			if (arc_start_angle > arc_end_angle)
			{
				std::swap(arc_start_angle, arc_end_angle);
			}

			const float delta_angle = 0.1f;

			avg_grays = cv::Mat::zeros(cv::Size(cvRound((arc_end_angle - arc_start_angle) / delta_angle), 1), CV_32F);

#pragma omp parallel for
			for (int i = 0; i != cvRound((arc_end_angle - arc_start_angle) / delta_angle); ++i)
			{
				// calculate avarge gray value along the current perpendicular line
				float avg = 0.0f;
				float arc_curr_angle = arc_start_angle + i * delta_angle;

				for (int j = 0; j != static_cast<int>((m_handle.radius - m_handle.annulus_radius) / step); ++j)
				{
					cv::Point2f curr_perpend_pt = cv::Point2f(
						m_handle.center.x + (m_handle.annulus_radius + j * step) * cos(-CV_PI / 180.0 * arc_curr_angle),
						m_handle.center.y + (m_handle.annulus_radius + j * step) * sin(-CV_PI / 180.0 * arc_curr_angle)
					);

					auto&& inter_point = interpolatePoint(curr_perpend_pt);	// interpolation
					avg += gray.at<uchar>(inter_point.y, inter_point.x);
				}

				avg_grays.at<float>(0, i) = (avg / (m_handle.radius - m_handle.annulus_radius) / step);
				center_pts.emplace_back(
					m_handle.center.x + 0.5 * (m_handle.annulus_radius + m_handle.radius) * cos(-CV_PI / 180.0 * arc_curr_angle),
					m_handle.center.y + 0.5 * (m_handle.annulus_radius + m_handle.radius) * sin(-CV_PI / 180.0 * arc_curr_angle)
				);
			}
		}

		else if (m_handle.type == MeasureHandleType::RECTANGLE)
		{
			// limit rotated rectangle phi in range [-90.0, 90.0]
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
			if (abs(limit_phi) != 90.0f)	// if not up-left rectangle, initialize
			{
				slope = tan(-CV_PI / 180.0f * limit_phi);

				if (limit_phi != 0.0f)
				{
					vertical_slope = (limit_phi < 0)
						? tan(-CV_PI / 180.0f * (limit_phi + 90.0))
						: tan(-CV_PI / 180.0f * (limit_phi - 90.0));
				}
			}

#pragma omp parallel for
			for (int i = 0; i != cvRound(2.0 * m_handle.length1); ++i)
			{
				cv::Point2f curr_profile_pt;
				// initialize current profile point
				if (abs(limit_phi) != 90.0f)
				{
					curr_profile_pt = getCurrPoint(
						profile_line.first, slope,
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

					// initialize points of perpendicular line
					if (abs(limit_phi) != 90.0f && limit_phi != 0.0f)
					{
						start_perpend_x =
							curr_profile_pt.x - 1.0 * m_handle.length2
							* sin(CV_PI / 180.0f * abs(limit_phi));

						curr_perpend_pt =
							getCurrPoint(curr_profile_pt, vertical_slope,
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

					const auto& inter_point = interpolatePoint(curr_perpend_pt);

					// check if is legal point
					if (inter_point.x < 0 || inter_point.y < 0)
					{
						cerr << "\nIllegal Postion...\nSkip Current Point...\n\n" << endl;
						++skip_num;
						continue;
					}
					//cv::namedWindow("gray", cv::WINDOW_FREERATIO);
					//cv::circle(gray, curr_perpend_pt, 3, cv::Scalar(0, 0, 0), -1);

					avg += gray.at<uchar>(inter_point.y, inter_point.x);
				}
				//cv::imshow("gray", gray);
				//cv::waitKey(1);
				avg_grays.at<float>(0, i) = static_cast<float>(avg / (m_handle.length2 * 2 / step - skip_num));
				center_pts.emplace_back(curr_profile_pt);
			}
		}

		cv::destroyAllWindows();
		return true;
	}

	double edgeSubPixel(const cv::Mat& deriv, double& offset, int local_max_ind)
	{
		if (local_max_ind == 0 || local_max_ind == deriv.cols - 1)	// neet at least 3 points
		{
			offset = local_max_ind;
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
		if (abs(offset - local_max_ind) >= 1.0)
			offset = local_max_ind;

		return (4.0 * coeffs.at<float>(2, 0) * coeffs.at<float>(0, 0) -
			coeffs.at<float>(1, 0) * coeffs.at<float>(1, 0)) / (4.0 * coeffs.at<float>(2, 0));
	}

	bool measurePos(const cv::Mat& src, const MeasureHandle& m_handle, std::vector<cv::Point2f>& edge_points,
		std::vector<double>& amplitudes, std::vector<double>& distances, float threshold, int ksize, float sigma,
		Transition transition, Select select, Interpolation inter)
	{
		static double timer_total = 0.0;// start calculating the time
		timer_total = cv::getTickCount();

		/**** Calculate Avarage Pixel Value ****/

		std::vector<cv::Point2f> center_pts;
		cv::Mat avg_grays;
		perpendAvgGrays(src, avg_grays, center_pts, m_handle);

		/**** Gaussian Blur & Calculate 1st Derivation ****/

		cv::Mat deriv1 = avg_grays;	// alias of avg_grays
		GaussianBlur(avg_grays, avg_grays, ksize, sigma);
		differentiate1st(avg_grays, sigma, deriv1);


		/**** Points Filter ****/

		constexpr int MAX_CONSEC_NUM = 4;	// how to set this value ?
		std::vector<size_t> candidate_ids;	// index of 1st derivation greater than threshold,
		// from left to right.

		edge_points.clear();	// clear useless data
		amplitudes.clear();

		switch (transition)
		{
		case Transition::ALL:
			for (size_t col = 0; col != deriv1.cols; ++col)
			{
				const auto& deriv_val = deriv1.at<float>(0, col);
				if (abs(deriv_val) > threshold)
				{
					candidate_ids.push_back(col);
				}
			}
			break;

		case Transition::NEGATIVE:
			for (size_t col = 0; col != deriv1.cols; ++col)
			{
				const auto& deriv_val = deriv1.at<float>(0, col);
				if (deriv_val < 0 && abs(deriv_val) > threshold)
				{
					candidate_ids.push_back(col);
				}
			}
			break;

		case Transition::POSITIVE:
			for (size_t col = 0; col != deriv1.cols; ++col)
			{
				const auto& deriv_val = deriv1.at<float>(0, col);
				if (deriv_val > 0 && abs(deriv_val) > threshold)
				{
					candidate_ids.push_back(col);
				}
			}
			break;

		default:
			break;
		}

		if (candidate_ids.empty())
		{
			cerr << "No Straight Edge Point Detected...\n\n" << endl;
			return false;
		}


		/**** Select Interesting Points ****/

		if (candidate_ids.size() >= 2)
		{
			std::vector<std::vector<size_t>> parts_inds;
			std::vector<size_t> parts = { candidate_ids[0] };	// each partition
			auto profile_line = getProfileLine(m_handle);

			for (size_t i = 1; i != candidate_ids.size(); ++i)
			{
				const int& pre_col = candidate_ids[i - 1];
				const int& curr_col = candidate_ids[i];
				const float& pre_amplitude = deriv1.at<float>(0, pre_col);
				const float& curr_amplitude = deriv1.at<float>(0, curr_col);

				/** Condition **
				*
				* 1. every partition size is less than MAX_CONSEC_NUM
				* 2. column distance is less than MAX_CONSEC_NUM
				* 3. current point and next point both have the same sign
				*/
				if (parts.size() <= MAX_CONSEC_NUM &&
					curr_col - pre_col < MAX_CONSEC_NUM &&
					(pre_amplitude <= 0.0 && curr_amplitude <= 0.0 ||
						pre_amplitude >= 0.0 && curr_amplitude >= 0.0))
				{
					parts.push_back(curr_col);
				}
				else
				{
					parts_inds.push_back(parts);
					parts.clear();
					parts.push_back(curr_col);
				}
			}

			// push last partition
			if (!parts.empty())
			{
				parts_inds.push_back(parts);
			}

			// search local maximum of each partition
			for (size_t i = 0; i != parts_inds.size(); ++i)
			{
				const auto& partition = parts_inds[i];
				float max_amp = 0.0f;
				int local_max_ind = -1;

				for (const auto& ind : partition)
				{
					const float& curr_amp = abs(deriv1.at<float>(0, ind));
					if (curr_amp >= max_amp)
					{
						max_amp = curr_amp;
						local_max_ind = ind;
					}
				}

				cv::Point2f edge_pt;
				double offset = 0.0;
				double amp = edgeSubPixel(deriv1, offset, local_max_ind);

				if (profile_line.first.x != profile_line.second.x)
				{
					double angle = atan(
						(profile_line.first.y - profile_line.second.y) /
						(profile_line.first.x - profile_line.second.x)
					);
					edge_pt = cv::Point2f(
						profile_line.first.x + offset * cos(angle),
						profile_line.first.y + offset * sin(angle)
					);
				}
				else
				{
					edge_pt = cv::Point2f(
						profile_line.first.x,
						profile_line.first.y + offset
					);
				}
				edge_points.emplace_back(edge_pt);
				amplitudes.push_back(amp);
			}
		}
		else if (candidate_ids.size() == 1)
		{
			edge_points.emplace_back(center_pts[candidate_ids[0]]);
			amplitudes.push_back(deriv1.at<float>(0, candidate_ids[0]));
		}

		cv::Point2f temp_pt;
		float temp_amp = 0.0f;

		switch (select)
		{
		case Select::ALL:
			distances.clear();
			if (edge_points.size() >= 2)
			{
				for (size_t k = 1; k != edge_points.size(); ++k)
				{
					const auto& curr_pt = edge_points[k];
					const auto& pre_pt = edge_points[k - 1];

					const double dist = sqrt(
						pow(curr_pt.x - pre_pt.x, 2.0) + pow(curr_pt.y - pre_pt.y, 2.0)
					);
					distances.push_back(dist);
				}
			}

			break;

		case Select::FIRST:
			temp_pt = edge_points.front();
			temp_amp = amplitudes.front();

			edge_points.clear();
			amplitudes.clear();

			edge_points.emplace_back(temp_pt);
			amplitudes.push_back(temp_amp);

			distances.clear();	// no distance
			break;

		case Select::LAST:
			temp_pt = edge_points.back();
			temp_amp = amplitudes.back();

			edge_points.clear();
			amplitudes.clear();

			edge_points.emplace_back(temp_pt);
			amplitudes.push_back(temp_amp);

			distances.clear();	// no distance
			break;

		default:
			break;
		}
		timer = (cv::getTickCount() - timer) / cv::getTickFrequency();
		timer_total = (cv::getTickCount() - timer_total) / cv::getTickFrequency();

		cout << "\n----- Measure Pos Consume: "
			<< timer * 1000.0 << " ms\n" << endl;

		cout << "----- Total Process Consume: "
			<< timer_total * 1000.0 << " ms \n" << endl;

		cout << "Amplitude is:\n";
		for (const auto& amp : amplitudes)
			cout << amp << " ";
		cout << "\nDistance is:\n";
		for (const auto& dist : distances)
			cout << dist << " ";
		cout << "\nEdge Point is:\n";
		for (const auto& pt : edge_points)
			cout << pt << " ";
		cout << endl;

		// show and print result
		bool display_result = false;
		if (display_result)
		{
			cv::Mat canvas = src.clone();

			for (const auto& pt : edge_points)
			{
				int b = rand() % 256, g = rand() % 256, r = rand() % 256;
				cv::circle(canvas, pt, 5, cv::Scalar(b, g, r), -1);
			}
			cv::namedWindow("Edge Points", cv::WINDOW_FREERATIO);
			cv::imshow("Edge Points", canvas);
			cv::waitKey(0);
		}

		return true;
	}

	static void linspace(const float start, const float end, const int num_steps, cv::Mat& dst)
	{
		dst = cv::Mat::zeros(cv::Size(num_steps, 1), CV_32F);

		float step = (end - start) / num_steps;

		for (int i = 0; i != num_steps; ++i)
		{
			dst.ptr<float>()[i] = start + step * i;
		}
	}

	static void meshgrid(cv::Mat& X, cv::Mat& Y)
	{
		X = cv::repeat(X, Y.cols, 1);
		Y = cv::repeat(Y.t(), 1, X.cols);
	}

	void ellipseEdgeDetect(const cv::Mat& src, const cv::Point2f& flatten_center, cv::Mat& dst, std::vector<cv::Point2f>& ellip_points)
	{
		//double timer = cv::getTickCount();

		/**** Flatten Image ****/

		cv::Mat gray;
		if (src.channels() == 3)
		{
			cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
		}
		else
		{
			gray = src;
		}

		int sample_radius = 0;
		if (flatten_center.x <= src.cols / 2 && flatten_center.y <= src.rows / 2)
		{
			sample_radius = cvCeil(std::sqrt(std::pow(flatten_center.x, 2.0) + std::pow(flatten_center.y, 2.0)));
		}
		else if (flatten_center.x >= src.cols / 2 && flatten_center.y <= src.rows / 2)
		{
			sample_radius = cvCeil(std::sqrt(std::pow(src.cols - flatten_center.x, 2.0) + std::pow(flatten_center.y, 2.0)));
		}
		else if (flatten_center.x <= src.cols / 2 && flatten_center.y >= src.rows / 2)
		{
			sample_radius = cvCeil(std::sqrt(std::pow(flatten_center.x, 2.0) + std::pow(src.rows - flatten_center.y, 2.0)));
		}
		else
		{
			sample_radius = cvCeil(std::sqrt(std::pow(src.cols - flatten_center.x, 2.0) + std::pow(src.rows - flatten_center.y, 2.0)));
		}

		const int num_sample_rad = sample_radius;	// sample number of raidus
		const int num_sample_phi = 2 * num_sample_rad;	// sample number of angle

		cv::Mat Phi, Rad;
		linspace(0.0f, CV_2PI, num_sample_phi, Phi);
		linspace(sample_radius, 0.0f, num_sample_rad, Rad);
		meshgrid(Phi, Rad);

		cv::Mat map_x = cv::Mat(Phi.size(), CV_32F);
		cv::Mat map_y = cv::Mat(Phi.size(), CV_32F);

		cv::parallel_for_(cv::Range(0, Phi.rows), [&](const cv::Range& range)	// OpenMP parallel
			{
				for (int row = range.start; row != range.end; ++row)
				{
					for (int col = 0; col != Phi.cols; ++col)
					{
						map_x.ptr<float>(row)[col] =
							cos(Phi.ptr<float>(row)[col]) * Rad.ptr<float>(row)[col] + flatten_center.x;

						map_y.ptr<float>(row)[col] =
							sin(Phi.ptr<float>(row)[col]) * Rad.ptr<float>(row)[col] + flatten_center.y;
					}
				}
			});

		cv::remap(gray, dst, map_x, map_y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 255);	// straight the ellipse

		//timer = (cv::getTickCount() - timer) / cv::getTickFrequency();
		//cout << "Flatten Image Consume: " << timer * 1000.0 << " ms\n" << endl;


		/**** Dynamic Programming to Search For Global Maximum Point ****/

		cv::Mat temp;
		int range_start = 0, range_end = dst.rows;	// to limit the range of searching
		//timer = cv::getTickCount();

		cv::copyMakeBorder(dst, temp, 1, 1, 0, 0, cv::BORDER_CONSTANT, 255);
		temp.convertTo(temp, CV_32F, -1, 255);		// reverse image pixel to 255 - pixelVal

		cv::Mat dp = cv::Mat::zeros(temp.rows, 1, CV_32F);
		cv::Mat_<int> index_map(dst.rows, dst.cols - 1, 0);

		for (int i = dst.cols - 1; i > 0; --i)
		{
			if (i == static_cast<int>(0.75 * dst.cols))
			{
				int local_max = argmax<float>((float*)dp.data, dp.rows);
				range_start = (local_max - dp.rows / 10 > 0) ? (local_max - dp.rows / 10) : 0;
				range_end = (local_max + dp.rows / 10 < temp.rows) ? (local_max + dp.rows / 10) : dp.rows - 1;
			}

			cv::Mat_<float> node_statue = cv::Mat_<float>::zeros(dst.rows, 1);
			cv::Mat_<int> road_index(dst.rows, 1, 0);

			cv::parallel_for_(cv::Range(0, dst.rows), [&](const cv::Range& range)
				{
					for (int j = range.start; j < range.end; ++j)
					{
						if (j < range_start || j > range_end)
							continue;

						cv::Mat cache = temp.rowRange(j, j + 3).colRange(i, i + 1) + dp.rowRange(j, j + 3);
						int local_max_ind = argmax((float*)cache.data, 3);
						node_statue.ptr<float>()[j] = cache.ptr<float>()[local_max_ind];
						if (local_max_ind + j - 1 < 0)
						{
							road_index.ptr<int>()[j] = 0;
						}
						else
						{
							road_index.ptr<int>()[j] = local_max_ind + j - 1;
						}
					}
				});
			node_statue.copyTo(dp.rowRange(1, dst.rows + 1));
			road_index.copyTo(index_map.colRange(i - 1, i));
		}

		//timer = (cv::getTickCount() - timer) / cv::getTickFrequency();
		//cout << "Dynamic Program: " << timer * 1000.0 << " ms\n" << endl;

		cv::Mat start = temp.rowRange(1, dst.rows + 1).colRange(0, 1) + dp.rowRange(1, dst.rows + 1);
		int bias = 0;
		int index = argmax(((float*)start.data), dst.rows - 0);	// global maxium point's index

		//std::vector<cv::Point> p;
		//p.resize(dst.cols);
		//p[0] = cv::Point(0, index);
		//cv::Mat drawImg;
		//dst.convertTo(drawImg, CV_8UC1);
		//cv::cvtColor(drawImg, drawImg, cv::COLOR_GRAY2BGR);
		//for (int i = 1; i < p.size(); i++)
		//{
		//	int py = index_map.ptr<int>(index)[i];
		//	p[i] = cv::Point(i, py);
		//	index = py;
		//	cv::circle(drawImg, p[i], 1, cv::Scalar(0, 0, 255));
		//}

		ellip_points.clear();
		ellip_points.resize(dst.cols);
		ellip_points[0] = cv::Point2f(map_x.ptr<float>(index)[0], map_y.ptr<float>(index)[0]);
		//timer = cv::getTickCount();
		for (int i = 1; i != ellip_points.size(); ++i)
		{
			int py = index_map.ptr<int>(index)[i];

			// Coordinate Conversion, dst(X, Y) = src(map_x(X, Y), map_y(X, Y))
			ellip_points[i] = cv::Point2f(map_x.ptr<float>(py)[i], map_y.ptr<float>(py)[i]);
			index = py;
		}
		//timer = (cv::getTickCount() - timer) / cv::getTickFrequency();
		//cout << "Coordinate Conversion Consume: " << timer * 1000.0 << " ms\n" << endl;
	}
}// namespace edgedet
