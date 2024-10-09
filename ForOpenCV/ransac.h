#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <random>
#include <ctime>
#include <vector>

static std::mt19937 rng;
static std::uniform_real_distribution<float> udist(0.f, 1.f);
static std::normal_distribution<float> ndist(0.f, 10.f);
const int WIDTH = 640;
const int HEIGHT = 480;

class Ransac
{
public:
    static void init_random();
    static float random_number();
    static float random_noise();
    static std::vector<cv::Point2f> generateLineData(int N, cv::Point2f p, cv::Point2f dir, float inlier_ratio);
    static std::vector<cv::Point2f> generateCircleData(int N, cv::Point2f p, float radius, float inlier_ratio);
    static cv::Point3f RANSAC_Line(std::vector<cv::Point2f>& points, int max_iterations, float sigma);
    static cv::Point3f RANSAC_Circle(std::vector<cv::Point2f>& points, int max_iterations, float sigma);
    static int lining();
private:
    static cv::Point3f fitCircle(std::vector <cv::Point2f>& points);
    static cv::Point3f fitCircle(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3);
    static cv::Point3f fitLine(std::vector<cv::Point2f>& points);
    static cv::Point3f fitLine(cv::Point2f p1, cv::Point2f p2);
    static float distance_from_line(cv::Point3f& l, cv::Point2f p);
    static float distance_from_circle(cv::Point3f& params, cv::Point2f p);
    static int iterationNumber(float confidence, int sampleSize, int inliers, int N);
};

