#include "evaluation.hpp"

#include <fstream>
#include <sstream>
#include <functional>
#include <iomanip>
#include <stdexcept>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

void Score::add(const Score& other) {
    tp += other.tp;
    fp += other.fp;
    fn += other.fn;
}

double Score::precision() const {
    return tp + fp > 0
        ? static_cast<double>(tp) / (tp + fp)
        : 0.0;
}

double Score::recall() const {
    return tp + fn > 0
        ? static_cast<double>(tp) / (tp + fn)
        : 0.0;
}

double Score::f1() const {
    return 2 * tp + fp + fn > 0
        ? 2.0 * tp / (2 * tp + fp + fn)
        : 0.0;
}

static int classIndex(const string& type) {
    const vector<string> names = {
        "open", "short", "mousebite",
        "spur", "copper", "pin-hole"
    };

    for (int i = 0; i < 6; ++i) {
        if (type == names[i]) {
            return i;
        }
    }

    return 6;  // unknown
}

vector<Detection> readGroundTruth(const fs::path& path) {
    ifstream input(path);

    if (!input) {
        throw runtime_error(
            "Could not open annotation: " + path.string()
        );
    }

    const vector<string> classNames = {
        "background",
        "open",
        "short",
        "mousebite",
        "spur",
        "copper",
        "pin-hole"
    };

    vector<Detection> groundTruth;
    string line;

    while (getline(input, line)) {
        if (line.find_first_not_of(" \t\r") == string::npos) {
            continue;
        }

        istringstream row(line);

        int x1, y1, x2, y2, classId;

        if (!(row >> x1 >> y1 >> x2 >> y2 >> classId)) {
            throw runtime_error(
                "Invalid annotation format: " + path.string()
            );
        }

        if (x1 < 0 || y1 < 0 || x2 < x1 || y2 < y1 ||
            classId < 1 || classId > 6) {
            throw runtime_error(
                "Invalid annotation values: " + path.string()
            );
        }

        // Treat the bottom-right pixel as inclusive.
        Rect box(x1, y1, x2 - x1 + 1, y2 - y1 + 1);

        groundTruth.push_back({box, classNames[classId]});
    }

    return groundTruth;
}

double calculateIoU(const Rect& a, const Rect& b) {
    double intersection = (a & b).area();

    double unionArea =
        static_cast<double>(a.area()) +
        static_cast<double>(b.area()) -
        intersection;

    return unionArea > 0.0
        ? intersection / unionArea
        : 0.0;
}

Score evaluate(
    const vector<Detection>& predictions,
    const vector<Detection>& groundTruth,
    bool requireClass
) {
    // Record which ground-truth boxes each prediction can match.
    vector<vector<int>> candidates(predictions.size());

    for (size_t p = 0; p < predictions.size(); ++p) {
        for (size_t g = 0; g < groundTruth.size(); ++g) {
            if (requireClass &&
                predictions[p].type != groundTruth[g].type) {
                continue;
            }

            double iou = calculateIoU(
                predictions[p].box,
                groundTruth[g].box
            );

            if (iou > 0.33) {
                candidates[p].push_back(static_cast<int>(g));
            }
        }
    }

    // owner[g] is the prediction assigned to ground-truth box g.
    // -1 means the ground-truth box is unmatched.
    vector<int> owner(groundTruth.size(), -1);

    // Allow reassignment to obtain the maximum number of matches.
    function<bool(int, vector<bool>&)> findMatch =
        [&](int predictionIndex, vector<bool>& visited) -> bool {
            for (int g : candidates[predictionIndex]) {
                if (visited[g]) {
                    continue;
                }

                visited[g] = true;

                if (owner[g] == -1 ||
                    findMatch(owner[g], visited)) {
                    owner[g] = predictionIndex;
                    return true;
                }
            }

            return false;
        };

    int matched = 0;

    for (size_t p = 0; p < predictions.size(); ++p) {
        vector<bool> visited(groundTruth.size(), false);

        if (findMatch(static_cast<int>(p), visited)) {
            ++matched;
        }
    }

    Score score;
    score.tp = matched;
    score.fp = static_cast<int>(predictions.size()) - matched;
    score.fn = static_cast<int>(groundTruth.size()) - matched;

    return score;
}

void updateClassificationStats(
    const vector<Detection>& predictions,
    const vector<Detection>& groundTruth,
    ClassificationStats& stats,
    const string& imageId
) {
    vector<vector<int>> candidates(predictions.size());

    for (size_t p = 0; p < predictions.size(); ++p) {
        for (size_t g = 0; g < groundTruth.size(); ++g) {
            if (calculateIoU(
                    predictions[p].box,
                    groundTruth[g].box
                ) > 0.33) {
                candidates[p].push_back(static_cast<int>(g));
            }
        }
    }

    // owner[g] stores the prediction matched to ground-truth box g.
    vector<int> owner(groundTruth.size(), -1);

    function<bool(int, vector<bool>&)> findMatch =
        [&](int p, vector<bool>& visited) -> bool {
            for (int g : candidates[p]) {
                if (visited[g]) {
                    continue;
                }

                visited[g] = true;

                if (owner[g] == -1 || findMatch(owner[g], visited)) {
                    owner[g] = p;
                    return true;
                }
            }

            return false;
        };

    for (size_t p = 0; p < predictions.size(); ++p) {
        vector<bool> visited(groundTruth.size(), false);
        findMatch(static_cast<int>(p), visited);
    }

    int matchedHere = 0;

    for (size_t g = 0; g < groundTruth.size(); ++g) {
        int p = owner[g];

        if (p == -1) {
            continue;
        }

        int actual = classIndex(groundTruth[g].type);
        int predicted = classIndex(predictions[p].type);

        // Ground-truth labels must be one of the six valid classes.
        if (actual == 6) {
            throw runtime_error("Invalid ground-truth class");
        }

        ++stats.matrix[actual][predicted];
        ++stats.matched;
        ++matchedHere;

        if (actual == predicted) {
            ++stats.correct;
        } else if (!imageId.empty()) {
            const Detection& truth = groundTruth[g];
            const Detection& prediction = predictions[p];

            cout << "\n[Classification mismatch]\n"
                << "Image: " << imageId << '\n'
                << "Ground-truth index: " << g << '\n'
                << "Actual: " << truth.type << '\n'
                << "Predicted: " << prediction.type << '\n'
                << "Ground-truth box: "
                << truth.box.x << ", "
                << truth.box.y << ", "
                << truth.box.x + truth.box.width - 1 << ", "
                << truth.box.y + truth.box.height - 1 << '\n'
                << "Prediction box: "
                << prediction.box.x << ", "
                << prediction.box.y << ", "
                << prediction.box.x + prediction.box.width - 1 << ", "
                << prediction.box.y + prediction.box.height - 1 << '\n'
                << "IoU: "
                << calculateIoU(prediction.box, truth.box)
                << "\n\n";            
        }
    }

    stats.unmatchedGT +=
        static_cast<int>(groundTruth.size()) - matchedHere;

    stats.unmatchedPred +=
        static_cast<int>(predictions.size()) - matchedHere;
}

void printClassificationStats(
    ostream& out,
    const ClassificationStats& stats
) {
    const vector<string> names = {
        "open", "short", "mousebite",
        "spur", "copper", "pin-hole", "unknown"
    };

    out << "\nClassification on location-matched defects\n";
    out << "Matched: " << stats.matched << '\n';
    out << "Correct: " << stats.correct << '\n';

    if (stats.matched > 0) {
        double accuracy =
            static_cast<double>(stats.correct) / stats.matched;

        out << fixed << setprecision(4)
            << "Accuracy: " << accuracy << '\n';
    } else {
        out << "Accuracy: N/A (no matched defects)\n";
    }

    out << "Unmatched ground truth: " << stats.unmatchedGT << '\n';
    out << "Unmatched predictions: " << stats.unmatchedPred << '\n';

    out << "\nConfusion matrix\n";
    out << "Rows = actual, columns = predicted\n\n";

    out << setw(12) << "Actual";

    for (const string& name : names) {
        out << setw(12) << name;
    }

    out << '\n';

    for (int actual = 0; actual < 6; ++actual) {
        out << setw(12) << names[actual];

        for (int predicted = 0; predicted < 7; ++predicted) {
            out << setw(12) << stats.matrix[actual][predicted];
        }

        out << '\n';
    }
}

void printScore(
    ostream& out,
    const string& title,
    const Score& score
) {
    out << title << '\n'
        << "TP: " << score.tp
        << ", FP: " << score.fp
        << ", FN: " << score.fn << '\n'
        << fixed << setprecision(4)
        << "Precision: " << score.precision() << '\n'
        << "Recall: " << score.recall() << '\n'
        << "F1: " << score.f1() << "\n\n";
}

void printDetectionSummary(
    ostream& os,
    const Score& location,
    const Score& classification
) {
    os << "\nDetection Performance\n";
    os << string(58, '-') << '\n';

    os << left  << setw(18) << "Metric"
       << right << setw(20) << "Location Only"
       << right << setw(20) << "Location + Class"
       << '\n';

    os << string(58, '-') << '\n';

    os << left  << setw(18) << "TP"
       << right << setw(20) << location.tp
       << right << setw(20) << classification.tp
       << '\n';

    os << left  << setw(18) << "FP"
       << right << setw(20) << location.fp
       << right << setw(20) << classification.fp
       << '\n';

    os << left  << setw(18) << "FN"
       << right << setw(20) << location.fn
       << right << setw(20) << classification.fn
       << '\n';

    os << fixed << setprecision(4);

    os << left  << setw(18) << "Precision"
       << right << setw(20) << location.precision()
       << right << setw(20) << classification.precision()
       << '\n';

    os << left  << setw(18) << "Recall"
       << right << setw(20) << location.recall()
       << right << setw(20) << classification.recall()
       << '\n';

    os << left  << setw(18) << "F1-score"
       << right << setw(20) << location.f1()
       << right << setw(20) << classification.f1()
       << '\n';

    os << string(58, '-') << "\n\n";
}