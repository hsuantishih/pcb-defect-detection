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

static void printBatchSummary(
    ostream& os,
    const string& dataset,
    int attempted,
    int succeeded,
    int failed,
    const Score& totalLocation,
    const Score& totalClass,
    const ClassificationStats& classificationStats
);

int main(int argc, char* argv[]) {
    const fs::path dataRoot = "./DeepPCB/PCBData";
    const fs::path outputRoot = "./output/batch";

    string dataset = "trainval";
    int maxImages = -1;
    bool saveImages = false;

    if (argc > 1) {
        dataset = argv[1];
    }

    if (argc > 2) {
        maxImages = stoi(argv[2]);
    }

    if (argc > 3) {
        saveImages = string(argv[3]) == "1";
    }


    ifstream list(dataRoot / (dataset + ".txt"));

    if (!list.is_open()) {
        cerr << "Could not open the dataset list\n";
        return 1;
    }

    fs::create_directories(outputRoot / "predictions");
    fs::create_directories(outputRoot / "images");

    ofstream failures(outputRoot / "failures.txt");

    if (!failures) {
        cerr << "Could not create the error log\n";
        return 1;
    }

    string imageRelative;
    string annotationRelative;

    int attempted = 0;
    int succeeded = 0;
    int failed = 0;

    Score totalLocation;
    Score totalClass;
    ClassificationStats classificationStats;

    while ((maxImages < 0 || attempted < maxImages) &&
           (list >> imageRelative >> annotationRelative)) {
        ++attempted;

        // Example: group20085/20085/20085000.jpg
        fs::path basePath = dataRoot / imageRelative;

        // stem() extracts the filename without its extension: 20085000
        string imageId = basePath.stem().string();

        fs::path tempPath =
            basePath.parent_path() / (imageId + "_temp.jpg");

        fs::path testPath =
            basePath.parent_path() / (imageId + "_test.jpg");

        try {
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

            // Process one image pair and collect all predictions.
            vector<Detection> predictions =
                detectDefects(tempImage, testImage);

            vector<Detection> groundTruth =
                readGroundTruth(dataRoot / annotationRelative);

            // Evaluate location only.
            Score locationScore =
                evaluate(predictions, groundTruth, false);

            // Evaluate both location and class.
            Score classScore =
                evaluate(predictions, groundTruth, true);

            fs::path predictionPath =
                outputRoot / "predictions" / (imageId + ".txt");

            ofstream output(predictionPath);

            if (!output) {
                throw runtime_error("Could not create the prediction file");
            }

            if (saveImages) {
                Mat result;
                cvtColor(testImage, result, COLOR_GRAY2BGR);

                // Draw prediction boxes
                for (const Detection& detection : predictions) {
                    const Rect& box = detection.box;
                    rectangle(result, box, Scalar(0, 0, 255), 2);
                    
                    putText(
                        result,
                        detection.type,
                        Point(box.x, max(15, box.y - 5)),
                        FONT_HERSHEY_SIMPLEX,
                        0.45,
                        Scalar(0, 255, 0),
                        1
                    );
                }

                // Draw ground-truth boxes
                for (const Detection& truth : groundTruth) {
                    rectangle(result, truth.box, Scalar(0, 255, 0), 1);
                }

                fs::path resultPath = outputRoot / "images" / (imageId + ".png");

                // Save the result image
                if (!imwrite(resultPath.string(), result)) {
                    throw runtime_error("Failed to save the result image");
                }

                // Save images containing false negatives separately
                if (locationScore.fn > 0) {
                    fs::create_directories(outputRoot / "FN");

                    fs::path fnPath =
                        outputRoot / "FN" / (imageId + ".png");

                    imwrite(fnPath.string(), result);
                }
            }
            
            
            for (const Detection& detection : predictions) {
                const Rect& box = detection.box;

                // Development output format:
                // x1,y1,x2,y2,confidence,type
                output << box.x << ','
                       << box.y << ','
                       << box.x + box.width - 1 << ','
                       << box.y + box.height - 1 << ','
                       << "1.0" << ','
                       << detection.type << '\n';
            }

            output.close();

            if (!output) {
                throw runtime_error("Failed to write the prediction file");
            }

            // Update classification statistics and visualize FN
            updateClassificationStats(
                predictions,
                groundTruth,
                classificationStats
            );

            totalLocation.add(locationScore);
            totalClass.add(classScore);

            ++succeeded;
        }
        catch (const exception& error) {
            ++failed;

            cerr << imageId << ": " << error.what() << '\n';
            failures << imageId << ": " << error.what() << '\n';
        }
    }


    printBatchSummary(
        cout,
        dataset,
        attempted,
        succeeded,
        failed,
        totalLocation,
        totalClass,
        classificationStats
    );

    ofstream summary(outputRoot / "summary.txt");

    if (!summary) {
        cerr << "Could not create the summary file\n";
        return 1;
    }

    printBatchSummary(
        summary,
        dataset,
        attempted,
        succeeded,
        failed,
        totalLocation,
        totalClass,
        classificationStats
    );

    summary.close();

    if (!summary) {
        cerr << "Failed to write the summary file\n";
        return 1;
    }

    return failed > 0 ? 1 : 0;
}



static void printBatchSummary(
    ostream& os,
    const string& dataset,
    int attempted,
    int succeeded,
    int failed,
    const Score& totalLocation,
    const Score& totalClass,
    const ClassificationStats& classificationStats
) {
    os << "\n";
    os << "==================== Batch Evaluation ====================\n\n";

    os << left << setw(19) << "Dataset:"
       << dataset << '\n';

    os << left << setw(19) << "IoU threshold:"
       << "> 0.33" << '\n';

    os << left << setw(19) << "Images attempted:"
       << attempted << '\n';

    os << left << setw(19) << "Images succeeded:"
       << succeeded << '\n';

    os << left << setw(19) << "Images failed:"
       << failed << '\n';

    os << left << setw(19) << "Ground truth:"
       << totalLocation.tp + totalLocation.fn << '\n';

    os << left << setw(19) << "Predictions:"
       << totalLocation.tp + totalLocation.fp << '\n';

    printDetectionSummary(
        os,
        totalLocation,
        totalClass
    );

    printClassificationStats(
        os,
        classificationStats
    );

    os << '\n';
    os << string(58, '=') << '\n';
    os << "Metrics cover successfully processed images only.\n";
}