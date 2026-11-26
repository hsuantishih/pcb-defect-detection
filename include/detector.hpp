#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct Detection {
    cv::Rect box;
    std::string type;
};

std::vector<Detection> detectDefects(
    const cv::Mat& temp_image,
    const cv::Mat& test_image
);