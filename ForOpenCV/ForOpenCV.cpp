#include "ransac.h"


int main()
{
    Ransac::init_random();
    cv::Mat img = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

    float MIN_RADIUS = 10.f;
    float MAX_RADIUS = 100.f;
    float radius = Ransac::random_number() * (MAX_RADIUS - MIN_RADIUS) + MIN_RADIUS;
    // for it not to go out of the window boundaries
    cv::Point2f center = { Ransac::random_number()*(WIDTH - 2*radius) + radius, Ransac::random_number()*(HEIGHT - 2*radius) + radius };
    
    std::cout << center << " " << radius << std::endl;

    std::vector<cv::Point2f> points = Ransac::generateCircleData(1000, center, radius, 0.5f);

    for (int i = 0; i < points.size(); ++i)
    {
        cv::circle(img, points[i], 2, { 255, 255, 255 }, cv::FILLED);
    }

    cv::Point3f model_params = Ransac::RANSAC_Circle(points, 2000, 10.f);

    cv::circle(img, center, radius, { 0, 255, 0 }, 2);
    cv::circle(img, cv::Point2f(model_params.x, model_params.y), model_params.z, { 0,0,255 }, 3);

    cv::imshow("RANSAC", img);
    int key = cv::waitKey(0);

    return 0;
}
