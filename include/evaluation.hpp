#pragma once

#include "detector.hpp"

#include <filesystem>
#include <iostream>
#include <vector>
#include <string>

struct Score {
    int tp = 0;
    int fp = 0;
    int fn = 0;

    void add(const Score& other);

    double precision() const;
    double recall() const;
    double f1() const;
};

struct ClassificationStats {
    int matrix[6][7] = {};

    int matched = 0;
    int correct = 0;
    int unmatchedGT = 0;
    int unmatchedPred = 0;
};

std::vector<Detection> readGroundTruth(
    const std::filesystem::path& path
);

double calculateIoU(
    const cv::Rect& a,
    const cv::Rect& b
);

Score evaluate(
    const std::vector<Detection>& predictions,
    const std::vector<Detection>& groundTruth,
    bool requireClass
);

void updateClassificationStats(
    const std::vector<Detection>& predictions,
    const std::vector<Detection>& groundTruth,
    ClassificationStats& stats,
    const std::string& imageId = ""
);

void printClassificationStats(
    std::ostream& out,
    const ClassificationStats& stats
);

void printScore(
    std::ostream& out,
    const std::string& title,
    const Score& score
);