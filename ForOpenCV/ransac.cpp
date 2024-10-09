#include "ransac.h"

#pragma region rand

void Ransac::init_random()
{
    rng.seed(static_cast<long unsigned int>(time(0)));
}
float Ransac::random_number()
{
    return udist(rng);
}
float Ransac::random_noise()
{
    return ndist(rng);
}
#pragma endregion

std::vector<cv::Point2f> Ransac::generateCircleData(int N, cv::Point2f center, float radius, float inlier_ratio) {
    std::vector<cv::Point2f> ret;

    int N_inlier = N * inlier_ratio,
        N_outlier = N - N_inlier;

    for (int i = 0; i < N_inlier; ++i)
    {
        float angle = random_number() * CV_2PI;
        cv::Point2f q = center + cv::Point2f(cos(angle), sin(angle)) * radius;
        q.x += random_noise();
        q.y += random_noise();
        ret.push_back(q);
    }

    for (int i = 0; i < N_outlier; ++i)
    {
        cv::Point2f q;
        q.x = random_number() * WIDTH;
        q.y = random_number() * HEIGHT;
        ret.push_back(q);
    }

    return ret;
}

cv::Point3f Ransac::RANSAC_Circle(std::vector<cv::Point2f>& points, int max_iterations, float sigma)
{
    int best_inliers_nr = 0;
    cv::Point3f best_model;
    std::vector<cv::Point2f> best_inliers;
    std::cout << "New max iters: " << max_iterations << std::endl;
    int m = 3;
    for (int iter = 0; iter < max_iterations; ++iter)
    {
        int* ptIdxs = new int[m];
        for (int idx = 0; idx < m; idx++) {
            ptIdxs[idx] = random_number() * (points.size() - 1);
            bool isInPts;
            while (true) {
                isInPts = false;
                for (int ii = 0; ii < m; ii++) {
                    if (idx != ii)
                        if (ptIdxs[idx] == ptIdxs[ii])
                            isInPts = true;
                }
                if (isInPts)
                    ptIdxs[idx] = random_number() * (points.size() - 1);
                else
                    break;
            }
        }

        cv::Point3f current_model = fitCircle(points[ptIdxs[0]], points[ptIdxs[1]], points[ptIdxs[2]]);

        int inliers_nr = 0;
        std::vector<cv::Point2f> current_inliers;
        for (int i = 0; i < points.size(); ++i)
        {
            if (distance_from_circle(current_model, points[i]) < sigma)
            {
                current_inliers.push_back(points[i]);
                ++inliers_nr;
            }
        }

        if (inliers_nr > best_inliers_nr)
        {
            best_inliers_nr = inliers_nr;
            best_model = current_model;
            best_inliers = current_inliers;

            max_iterations = std::min(max_iterations, iterationNumber(0.95, 2, best_inliers_nr, points.size()));
        }
    }
    
    std::cout << best_model << std::endl;

    //!!!
    best_model = fitCircle(best_inliers);
    //!!!

    std::cout << fitCircle(best_inliers) << std::endl;

    return best_model;
}

cv::Point3f Ransac::fitCircle(std::vector <cv::Point2f>& points) {
    cv::Mat x;

    cv::Mat A = cv::Mat(points.size(), 3, CV_32F);
    cv::Mat b = cv::Mat(points.size(), 1, CV_32F);
    for (int i = 0; i < points.size(); ++i)
    {
        A.at<float>(i, 0) = points[i].x;
        A.at<float>(i, 1) = points[i].y;
        A.at<float>(i, 2) = 1.f;
        b.at<float>(i, 0) = points[i].x * points[i].x + points[i].y * points[i].y;
    }
    cv::Mat Aplus = (A.t() * A).inv() * A.t();
    x = Aplus * b;

    float x_c = 0.5 * x.at<float>(0, 0);
    float y_c = 0.5 * x.at<float>(1, 0);
    float r = sqrt(x.at<float>(2, 0) + x_c * x_c + y_c * y_c);

    cv::Point3f model = cv::Point3f(x_c, y_c, r);

    return model;
}

cv::Point3f Ransac::fitCircle(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3) {
    cv::Mat x;

    float data[9] = { p1.x,p1.y,1,p2.x,p2.y,1,p3.x,p3.y,1 };
    float bdata[3] = { p1.x * p1.x + p1.y * p1.y, p2.x * p2.x + p2.y * p2.y, p3.x * p3.x + p3.y * p3.y };
    cv::Mat A = cv::Mat(3, 3, CV_32F, data);
    cv::Mat b = cv::Mat(3, 1, CV_32F, bdata);
    //std::cout << A << std::endl;
    //std::cout << b << std::endl;

    cv::solve(A, b, x);
   
    //std::cout << x << std::endl;  
    float x_c = 0.5 * x.at<float>(0, 0);
    float y_c = 0.5 * x.at<float>(1, 0);
    float r = sqrt(x.at<float>(2, 0) + x_c * x_c + y_c * y_c);

    cv::Point3f model = cv::Point3f(x_c, y_c, r);
    //std::cout << ans << std::endl;

    return model;
}

float Ransac::distance_from_circle(cv::Point3f& params, cv::Point2f p) {
    return abs(sqrt((p.x - params.x) * (p.x - params.x) + (p.y - params.y) * (p.y - params.y)) - params.z);
}

int Ransac::iterationNumber(float confidence, int sampleSize, int inliers, int N)
{
    float a = std::log(1.0f - confidence);
    float ratio = (float)inliers / (float)N;
    float b = std::log(1.0f - std::pow(ratio, sampleSize));

    return std::ceil(a / b);
}


#pragma region line_only
std::vector<cv::Point2f> Ransac::generateLineData(int N, cv::Point2f p, cv::Point2f dir, float inlier_ratio)
{
    std::vector<cv::Point2f> ret;

    int N_inlier = N * inlier_ratio;
    int N_outlier = N - N_inlier;

    for (int i = 0; i < N_inlier; ++i)
    {
        float diag = sqrt(WIDTH * WIDTH + HEIGHT * HEIGHT);

        float t = random_number() * diag - diag / 2.f;
        std::cout << t << std::endl;
        cv::Point2f q = p + t * dir;
        q.x += random_noise();
        q.y += random_noise();

        ret.push_back(q);
    }

    for (int i = 0; i < N_outlier; ++i)
    {
        cv::Point2f q;
        q.x = random_number() * WIDTH;
        q.y = random_number() * HEIGHT;
        ret.push_back(q);
    }

    return ret;
}


cv::Point3f Ransac::RANSAC_Line(std::vector<cv::Point2f>& points, int max_iterations, float sigma)
{
    int best_inliers_nr = 0;
    cv::Point3f best_model;
    std::vector<cv::Point2f> best_inliers;
    std::cout << "New max iters: " << max_iterations << std::endl;
    for (int iter = 0; iter < max_iterations; ++iter)
    {
        // 1. Select m points randomly
        int idx1 = random_number() * (points.size() - 1);
        int idx2 = random_number() * (points.size() - 1);
        while (idx1 == idx2)
        {
            idx2 = random_number() * (points.size() - 1);
        }

        // 2. Fit a model on the sample
        cv::Point3f current_model = fitLine(points[idx1], points[idx2]);

        // 3. Calculating the distance of the points
        int inliers_nr = 0;
        std::vector<cv::Point2f> current_inliers;
        for (int i = 0; i < points.size(); ++i)
        {
            if (distance_from_line(current_model, points[i]) < sigma)
            {
                current_inliers.push_back(points[i]);
                ++inliers_nr;
            }
        }

        // 4. Save the best model so far
        if (inliers_nr > best_inliers_nr)
        {
            best_inliers_nr = inliers_nr;
            best_model = current_model;
            best_inliers = current_inliers;

            max_iterations = std::min(max_iterations, iterationNumber(0.95, 2, best_inliers_nr, points.size()));
            //std::cout << "New max iters: " << max_iterations << std::endl;
        }
    }

    best_model = fitLine(best_inliers);
    return best_model;
}

cv::Point3f Ransac::fitLine(std::vector<cv::Point2f>& points)
{
    cv::Mat X(points.size(), 3, CV_32F);
    for (int i = 0; i < points.size(); ++i)
    {
        X.at<float>(i, 0) = points[i].x;
        X.at<float>(i, 1) = points[i].y;
        X.at<float>(i, 2) = 1.f;
    }

    cv::Mat XtX = X.t() * X;

    cv::Mat evals, evecs;
    cv::eigen(XtX, evals, evecs);

    cv::Point3f l;
    l.x = evecs.at<float>(2, 0);
    l.y = evecs.at<float>(2, 1);
    l.z = evecs.at<float>(2, 2);

    // Normalization
    float n = sqrt(l.x * l.x + l.y * l.y);
    l /= n;

    return l;
}

cv::Point3f Ransac::fitLine(cv::Point2f p1, cv::Point2f p2)
{
    cv::Point2f dir = p2 - p1;
    dir /= sqrt(dir.x * dir.x + dir.y * dir.y); // normalization

    cv::Point3f l;
    l.x = -dir.y;
    l.y = dir.x;
    l.z = -l.x * p1.x - l.y * p1.y;

    return l;
}

float Ransac::distance_from_line(cv::Point3f& l, cv::Point2f p)
{
    return abs(l.x * p.x + l.y * p.y + l.z);
}

int Ransac::lining()
{
    Ransac::init_random();
    cv::Mat img = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

    cv::Point2f p = { HEIGHT / 2.f, WIDTH / 2.f };
    float alpha = Ransac::random_number() * CV_2PI;

    cv::Point2f dir;
    dir.x = cos(alpha);
    dir.y = sin(alpha);
    std::vector<cv::Point2f> points = Ransac::generateLineData(1000, p, dir, 0.5f);

    for (int i = 0; i < points.size(); ++i)
    {
        cv::circle(img, points[i], 2, { 255, 255, 255 }, cv::FILLED);
    }

    cv::Point2f p1, p2;
    p1 = p - 1000.f * dir;
    p2 = p + 1000.f * dir;
    cv::line(img, p1, p2, { 255, 0, 0 }, 2);

    cv::Point3f l = Ransac::RANSAC_Line(points, 2000, 10.f);
    //cv::Point3f l = fitLine(points);
    p1.x = 0;
    p1.y = -l.z / l.y;

    p2.x = WIDTH;
    p2.y = (-l.z - l.x * p2.x) / l.y;

    cv::line(img, p1, p2, { 0, 255, 0 }, 2);

    cv::imshow("RANSAC", img);
    int key = cv::waitKey(0);

    return 0;
}
#pragma endregion