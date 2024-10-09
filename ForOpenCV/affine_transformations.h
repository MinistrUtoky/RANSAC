#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

class Affine
{
public:
	static cv::Mat createTransformation(float angle, float tx, float ty, float sx, float sy, float skew_factor, float persp1, float persp2);
	static cv::Mat applyTransformation(cv::Mat& img, cv::Mat T, bool isPerspective, bool interpolated);
private:
	static cv::Vec2f newPoint(int i, int j, cv::Mat T, bool isPerspective);
	static float S(int x1, int y1, int x2, int y2, int x3, int y3) {
		return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0);
	}
	static bool withinTriangle(int x1, int y1, int x2, int y2, int x3, int y3, int pointX, int pointY, bool print=false) {
		double d = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
		double a = ((y2 - y3) * (pointX - x3) + (x3 - x2) * (pointY - y3)) / d;
		double b = ((y3 - y1) * (pointX - x3) + (x1 - x3) * (pointY - y3)) / d;
		double c = 1 - a - b;

		if (a >= 0 && b >= 0 && c >= 0) return true;
		else return false;
	}
};
