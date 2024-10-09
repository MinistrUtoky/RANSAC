#include "affine_transformations.h"

cv::Mat Affine::createTransformation(float angle, float tx, float ty, float sx, float sy, float skew_factor, float persp1, float persp2)
{
	// Rotation
	cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
	R.at<float>(0, 0) = cos(angle);
	R.at<float>(0, 1) = -sin(angle);
	R.at<float>(1, 0) = sin(angle);
	R.at<float>(1, 1) = cos(angle);

	// Translation
	cv::Mat t = cv::Mat::eye(3, 3, CV_32F);
	t.at<float>(0, 2) = tx;
	t.at<float>(1, 2) = ty;

	// Scaling
	cv::Mat Scale = cv::Mat::eye(3, 3, CV_32F);
	Scale.at<float>(0, 0) = sx;
	Scale.at<float>(1, 1) = sy;

	// Skewing
	cv::Mat Skew = cv::Mat::eye(3, 3, CV_32F);
	Skew.at<float>(0, 1) = skew_factor;

	cv::Mat T = t * R * Skew * Scale;

	// Perspective transformation
	T.at<float>(2, 0) = persp1;
	T.at<float>(2, 1) = persp2;

	return T;
}

cv::Vec2f Affine::newPoint(int i, int j, cv::Mat T, bool isPerspective)
{
	cv::Mat p(3, 1, CV_32F);
	p.at<float>(0, 0) = i;
	p.at<float>(1, 0) = j;
	p.at<float>(2, 0) = 1.f;

	cv::Mat newP = T * p;
	float newZ = 1.f;
	if (isPerspective)
		newZ = newP.at<float>(2, 0);

	float newX, newY;
	newX = newP.at<float>(0, 0) / newZ;
	newY = newP.at<float>(1, 0) / newZ;
	return cv::Vec2f(newX, newY);
}

cv::Mat Affine::applyTransformation(cv::Mat& img, cv::Mat T, bool isPerspective, bool interpolated=false)
{
	cv::Mat newImg = cv::Mat::zeros(img.size(), CV_8UC3);
	int HEIGHT = img.size().height;
	int WIDTH = img.size().width;

	for (int i = 0; i < HEIGHT; ++i)
		for (int j = 0; j < WIDTH; ++j)
		{
			cv::Vec2f newP = Affine::newPoint(i, j, T, isPerspective);
			int newX = newP[0], newY = newP[1];

			if (0 <= newX && newX < HEIGHT && 0 <= newY && newY < WIDTH)
				newImg.at<cv::Vec3b>(newX, newY) = img.at<cv::Vec3b>(i, j);
		}

	cv::Vec2f leftUp = Affine::newPoint(0, 0, T, isPerspective);
	cv::Vec2f leftDown = Affine::newPoint(HEIGHT-1, 0, T, isPerspective);
	cv::Vec2f rightUp = Affine::newPoint(0, WIDTH-1, T, isPerspective);
	cv::Vec2f rightDown = Affine::newPoint(HEIGHT - 1, WIDTH-1, T, isPerspective);
	int x1 = leftUp[0], y1 = leftUp[1],
		x2 = leftDown[0], y2 = leftDown[1],
		x3 = rightUp[0], y3 = rightUp[1],
		x4 = rightDown[0], y4 = rightDown[1];

	for (int i = 0; i < HEIGHT; ++i) {
		cv::Vec3b lastNotedColor = cv::Vec3b(0, 0, 0);
		for (int j = 0; j < WIDTH; ++j)
			if (withinTriangle(leftUp[0], leftUp[1], leftDown[0], leftDown[1], rightUp[0], rightUp[1], i, j)
				|| withinTriangle(rightUp[0], rightUp[1], rightDown[0], rightDown[1], leftDown[0], leftDown[1], i, j)) {
				if (newImg.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) 
					lastNotedColor = newImg.at<cv::Vec3b>(i, j);
				else if (lastNotedColor != cv::Vec3b(0, 0, 0))
					newImg.at<cv::Vec3b>(i, j) = lastNotedColor;
			}
	}
	for (int j = 0; j < WIDTH; ++j) {
		cv::Vec3b lastNotedColor = cv::Vec3b(0, 0, 0);
		for (int i = 0; i < HEIGHT; ++i)
		{
			if (withinTriangle(leftUp[0], leftUp[1], leftDown[0], leftDown[1], rightUp[0], rightUp[1], i, j)
				|| withinTriangle(rightUp[0], rightUp[1], rightDown[0], rightDown[1], leftDown[0], leftDown[1], i, j)) {
				if (newImg.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) 
					lastNotedColor = newImg.at<cv::Vec3b>(i, j);
				else if (lastNotedColor != cv::Vec3b(0,0,0))
					newImg.at<cv::Vec3b>(i, j) = lastNotedColor;
			}
		}
	}

	return newImg;
}