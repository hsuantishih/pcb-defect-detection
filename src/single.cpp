#include "detector.hpp"
#include "evaluation.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    string groupId = "20085";
    string suffixId = "000";
    bool saveResults = false;
    bool saveImages = false;

    if (argc > 1) {
        groupId = argv[1];
    }

    if (argc > 2) {
        suffixId = argv[2];
    }

    if (argc > 3) {
        saveResults = string(argv[3]) == "1";
    }

    if (argc > 4) {
        saveImages = string(argv[4]) == "1";
    }

    // Change these three paths to inspect another sample.
    fs::path tempPath =
        "./DeepPCB/PCBData/group" + groupId + "/" + groupId + "/" + groupId + suffixId + "_temp.jpg";

    const fs::path testPath =
        "./DeepPCB/PCBData/group" + groupId + "/" + groupId + "/" + groupId + suffixId + "_test.jpg";

    const fs::path labelPath =
        "./DeepPCB/PCBData/group" + groupId + "/" + groupId + "_not/" + groupId + suffixId + ".txt";

    const string imageId = labelPath.stem().string();

    // Re-running the same sample overwrites its output files.
    const fs::path outputDir =
        fs::path("./output/single") / imageId;

    try {
        // 1. Load one image pair.
        Mat tempImage = imread(
            tempPath.string(), IMREAD_GRAYSCALE
        );

        Mat testImage = imread(
            testPath.string(), IMREAD_GRAYSCALE
        );

        if (tempImage.empty() || testImage.empty()) {
            throw runtime_error("Could not load the image pair");
        }

        if (tempImage.size() != testImage.size()) {
            throw runtime_error("Image dimensions do not match");
        }

        // 2. Detect defects and read annotations.
        vector<Detection> predictions =
            detectDefects(tempImage, testImage);

        vector<Detection> groundTruth =
            readGroundTruth(labelPath);

        // 3. Evaluate this sample.
        Score locationScore =
            evaluate(predictions, groundTruth, false);

        Score classScore =
            evaluate(predictions, groundTruth, true);

        ClassificationStats classificationStats;

        // Also prints classification mismatches to the terminal.
        updateClassificationStats(
            predictions,
            groundTruth,
            classificationStats,
            imageId
        );

        // 4. Draw predictions and annotations separately.
        Mat predictionView;
        cvtColor(testImage, predictionView, COLOR_GRAY2BGR);

        Mat groundTruthView = predictionView.clone();
        Mat overlayView = predictionView.clone();

        auto drawLabel = [](Mat& image,
                            const Detection& detection,
                            const string& text,
                            const Scalar& color) {
            rectangle(image, detection.box, color, 2);

            Point position(
                detection.box.x,
                max(16, detection.box.y - 5)
            );

            // Black outline improves text readability.
            putText(
                image, text, position,
                FONT_HERSHEY_SIMPLEX, 0.45,
                Scalar(0, 0, 0), 3, LINE_AA
            );

            putText(
                image, text, position,
                FONT_HERSHEY_SIMPLEX, 0.45,
                color, 1, LINE_AA
            );
        };

        for (size_t i = 0; i < predictions.size(); ++i) {
            const Detection& prediction = predictions[i];

            drawLabel(
                predictionView,
                prediction,
                "P" + to_string(i) + " " + prediction.type,
                Scalar(0, 0, 255)
            );

            rectangle(
                overlayView,
                prediction.box,
                Scalar(0, 0, 255),
                2
            );

            drawLabel(
                overlayView,
                prediction,
                "P" + to_string(i) + " " + prediction.type,
                Scalar(0, 0, 255)
            );
        }

        for (size_t i = 0; i < groundTruth.size(); ++i) {
            const Detection& truth = groundTruth[i];

            drawLabel(
                groundTruthView,
                truth,
                "G" + to_string(i) + " " + truth.type,
                Scalar(0, 255, 0)
            );

            rectangle(
                overlayView,
                truth.box,
                Scalar(0, 255, 0),
                1
            );

            drawLabel(
                overlayView,
                truth,
                "G" + to_string(i) + " " + truth.type,
                Scalar(0, 255, 0)
            );
        }

        // 5. Save evaluation results.
        if (saveResults || saveImages) {
            fs::create_directories(outputDir);
        }

        ofstream summaryFile;

        if (saveResults) {
            fs::create_directories(outputDir);
            summaryFile.open(outputDir / "summary.txt");

            if (!summaryFile) {
                throw runtime_error("Could not create summary file");
            }
        }

        ostream& out = saveResults ? summaryFile : cout;

        out << "Image: " << imageId << '\n'
                << "Local evaluation, IoU > 0.33\n\n";

        printScore(out, "Location only", locationScore);
        printScore(out, "Location and class", classScore);
        printClassificationStats(out, classificationStats);


        if (saveImages) {
            // 6. Save visualization images.
            if (!imwrite(
                    (outputDir / "predictions.png").string(),
                    predictionView)) {
                throw runtime_error("Could not save prediction image");
            }
    
            if (!imwrite(
                    (outputDir / "ground_truth.png").string(),
                    groundTruthView)) {
                throw runtime_error("Could not save ground-truth image");
            }
    
            if (!imwrite(
                    (outputDir / "overlay.png").string(),
                    overlayView)) {
                throw runtime_error("Could not save overlay image");
            }
    
            cout << "\nResults saved to: " << outputDir << '\n';
        }

        cout << "\nGreen boxes = ground truth, red boxes = predictions\n";

        namedWindow("Overlay", WINDOW_NORMAL);
        imshow("Overlay", overlayView);
        waitKey(0);

        return 0;
    }
    catch (const exception& error) {
        cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}