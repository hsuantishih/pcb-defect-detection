#include "detector.hpp"
#include "opencv2/imgproc.hpp"

#include <set>
#include <algorithm>
#include <filesystem>

using namespace std;
using namespace cv;

static set<int> getLabels(const Mat& labels, const Mat& mask) {
    set<int> ids;

    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            if (mask.at<uchar>(y, x) != 0) {
                int id = labels.at<int>(y, x);

                if (id > 0) {
                    ids.insert(id);
                }
            }
        }
    }

    return ids;
}

static string classifyDefect(
    const Mat& tempCopper,
    const Mat& testCopper,
    const Mat& defectMask,
    const Rect& box
) {
    const int padding = 20;

    Rect roi(
        box.x - padding,
        box.y - padding,
        box.width + padding * 2,
        box.height + padding * 2
    );

    // Ensure the ROI is within the image boundaries
    roi &= Rect(0, 0, tempCopper.cols, tempCopper.rows);

    // Extract the regions of interest
    Mat tempROI = tempCopper(roi);
    Mat testROI = testCopper(roi);
    Mat maskROI = defectMask(roi);

    // Separate the extra copper and missing copper areas
    Mat notTemp, notTest, extra, missing;

    bitwise_not(tempROI, notTemp);
    bitwise_not(testROI, notTest);

    bitwise_and(testROI, notTemp, extra);
    bitwise_and(tempROI, notTest, missing);

    bitwise_and(extra, maskROI, extra);
    bitwise_and(missing, maskROI, missing);

    int extraCount = countNonZero(extra);
    int missingCount = countNonZero(missing);

    if (extraCount == 0 && missingCount == 0) {
        return "unknown";
    }

    // Handle cases where both extra and missing copper are present
    if (extraCount > 0 && missingCount > 0) {
        double ratio =
            static_cast<double>(min(extraCount, missingCount)) /
            max(extraCount, missingCount);

        if (ratio > 0.25) {
            return "unknown";
        }
    }

    bool isExtra = extraCount > missingCount;
    Mat defect = isExtra ? extra : missing;

    // Create a ring around the defect area
    Mat expanded, inverseDefect, ring;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    dilate(defect, expanded, kernel);
    bitwise_not(defect, inverseDefect);
    bitwise_and(expanded, inverseDefect, ring);
    
    Mat tempLabels;
    connectedComponents(tempROI, tempLabels, 8, CV_32S);

    if (isExtra) {
        set<int> touched = getLabels(tempLabels, ring);

        if (touched.empty()) {
            return "copper";
        }

        if (touched.size() >= 2) {
            return "short";
        }

        return "spur";
    }

    // Check if the missing copper area touches the background
    Mat backgroundContact;
    bitwise_and(ring, notTemp, backgroundContact);

    bool touchesImageEdge =
        box.x == 0 ||
        box.y == 0 ||
        box.x + box.width == tempCopper.cols ||
        box.y + box.height == tempCopper.rows;

    if (countNonZero(backgroundContact) == 0 && !touchesImageEdge) {
        return "pin-hole";
    }

    Mat testLabels;
    Mat stats;
    Mat centroids;

    connectedComponentsWithStats(testROI, testLabels, stats, centroids, 8, CV_32S);

    set<int> originalIds = getLabels(tempLabels, defect);

    // Check if the same copper region in the template has become multiple copper regions around the missing copper area
    for (int originalId : originalIds) {
        Mat originalRegion, nearbyRegion;

        compare(tempLabels, originalId, originalRegion, CMP_EQ);
        bitwise_and(originalRegion, ring, nearbyRegion);

        set<int> remainingIds = getLabels(testLabels, nearbyRegion);

        const int minComponentArea = 5;
        set<int> validIds;

        for (int id : remainingIds) {
            int area = stats.at<int>(id, CC_STAT_AREA);

            if (area >= minComponentArea) {
                validIds.insert(id);
            }

            if (validIds.size() >= 2) {
                return "open";
            }
        }

    }

    return "mousebite";
}

static Mat getToleranceMask(
    const Mat& tempCopper,
    const Mat& testCopper,
    int tolerance
) {
    Mat kernel = getStructuringElement(
        MORPH_ELLIPSE,
        Size(2 * tolerance + 1, 2 * tolerance + 1)
    );

    // Expand the acceptable copper boundaries
    Mat dilatedTemp, dilatedTest;

    dilate(tempCopper, dilatedTemp, kernel);
    dilate(testCopper, dilatedTest, kernel);

    Mat notDilatedTemp, notDilatedTest;

    bitwise_not(dilatedTemp, notDilatedTemp);
    bitwise_not(dilatedTest, notDilatedTest);

    // Detect differences beyond the boundary tolerance
    Mat extra, missing;

    bitwise_and(testCopper, notDilatedTemp, extra);
    bitwise_and(tempCopper, notDilatedTest, missing);

    Mat toleranceMask;
    bitwise_or(extra, missing, toleranceMask);

    return toleranceMask;
}

vector<Detection> detectDefects(
    const Mat& temp_image,
    const Mat& test_image
) {
    Mat tempCopper, testCopper;

    threshold(temp_image, tempCopper, 128, 255, THRESH_BINARY_INV);
    threshold(test_image, testCopper, 128, 255, THRESH_BINARY_INV);

    // Difference
    Mat diff;
    absdiff(temp_image, test_image, diff);

    // Threshold
    Mat binary;
    threshold(diff, binary, 40, 255, THRESH_BINARY);

    const int tolerance = 3;

    Mat toleranceMask = getToleranceMask(tempCopper, testCopper, tolerance);

    Mat filtered;

    bitwise_and(binary, toleranceMask, filtered);

    // Morphological Opening
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat dst;
    morphologyEx(filtered, dst, MORPH_OPEN, kernel);

    // Debug images
    string debugDir = "./debug/";

    if (!filesystem::exists(debugDir)) {
        filesystem::create_directory(debugDir);
    }
    imwrite(debugDir + "01_testCopper.png", testCopper);
    imwrite(debugDir + "02_tempCopper.png", tempCopper);
    imwrite(debugDir + "03_diff.png", diff);
    imwrite(debugDir + "04_binary.png", binary);
    imwrite(debugDir + "05_toleranceMask.png", toleranceMask);
    imwrite(debugDir + "06_filtered.png", filtered);
    imwrite(debugDir + "07_opened.png", dst);

    // Find Contours
    vector<vector<Point>> contours;
    findContours(dst, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat result;
    cvtColor(test_image, result, COLOR_GRAY2BGR);

    vector<Detection> detections;

    for (size_t i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        
        if (area < 10) {
            continue;
        }
        
        Rect box = boundingRect(contours[i]);

        Mat defectMask = Mat::zeros(binary.size(), CV_8UC1);

        drawContours(
            defectMask,
            contours,
            static_cast<int>(i),
            Scalar(255),
            FILLED
        );

        // Restrict the defect mask to the binary image
        bitwise_and(defectMask, binary, defectMask);

        // Classify the defect based on the extracted regions
        string type = classifyDefect(
            tempCopper,
            testCopper,
            defectMask,
            box
        );

        const int boxPadding = 8;

        Rect outputBox(
            box.x - boxPadding,
            box.y - boxPadding,
            box.width + 2 * boxPadding,
            box.height + 2 * boxPadding
        );

        outputBox &= Rect(0, 0, test_image.cols, test_image.rows);

        detections.push_back({outputBox, type});
    }

    return detections;
}