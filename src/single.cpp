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

int main() {
    // Change these three paths to inspect another sample.
    const fs::path tempPath =
        "./DeepPCB/PCBData/group20085/20085/20085000_temp.jpg";

    const fs::path testPath =
        "./DeepPCB/PCBData/group20085/20085/20085000_test.jpg";

    const fs::path labelPath =
        "./DeepPCB/PCBData/group20085/20085_not/20085000.txt";

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

        cout << "\nImage: " << imageId << '\n'
             << "Predictions: " << predictions.size() << '\n'
             << "Ground truth: " << groundTruth.size() << "\n\n";

        printScore(cout, "Location only", locationScore);
        printScore(cout, "Location and class", classScore);
        printClassificationStats(cout, classificationStats);

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
        }

        // 5. Save predictions.
        fs::create_directories(outputDir);

        ofstream predictionFile(outputDir / "predictions.txt");

        if (!predictionFile) {
            throw runtime_error("Could not create prediction file");
        }

        for (const Detection& prediction : predictions) {
            const Rect& box = prediction.box;

            predictionFile
                << box.x << ','
                << box.y << ','
                << box.x + box.width - 1 << ','
                << box.y + box.height - 1 << ','
                << "1.0" << ','
                << prediction.type << '\n';
        }

        predictionFile.close();

        if (!predictionFile) {
            throw runtime_error("Failed to write predictions");
        }

        // 6. Save evaluation results.
        ofstream summary(outputDir / "summary.txt");

        if (!summary) {
            throw runtime_error("Could not create summary file");
        }

        summary << "Image: " << imageId << '\n'
                << "Local evaluation, IoU > 0.33\n\n";

        printScore(summary, "Location only", locationScore);
        printScore(summary, "Location and class", classScore);
        printClassificationStats(summary, classificationStats);

        summary.close();

        if (!summary) {
            throw runtime_error("Failed to write summary");
        }

        // 7. Save visualization images.
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
        cout << "Press any key in an image window to exit.\n";

        // // 8. Display results.
        // namedWindow("Predictions", WINDOW_NORMAL);
        // namedWindow("Ground truth", WINDOW_NORMAL);
        // namedWindow("Overlay", WINDOW_NORMAL);

        // imshow("Predictions", predictionView);
        // imshow("Ground truth", groundTruthView);
        // imshow("Overlay", overlayView);

        // waitKey(0);
        // destroyAllWindows();

        return 0;
    }
    catch (const exception& error) {
        cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}