// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  SentinelNN — From-Scratch Neural Network Text Classifier in Pure C++     ║
// ║  Multi-class sentiment analysis with TF-IDF + bigrams                     ║
// ║  Version 2.0                                                              ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <map>
#include <set>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <cassert>
#include <functional>

// ─────────────────────────────────────────────────────────────────────────────
// ANSI Color & Style Codes
// ─────────────────────────────────────────────────────────────────────────────
namespace Color {
    const std::string RESET   = "\033[0m";
    const std::string BOLD    = "\033[1m";
    const std::string DIM     = "\033[2m";
    const std::string ITALIC  = "\033[3m";
    const std::string UNDER   = "\033[4m";

    const std::string RED     = "\033[31m";
    const std::string GREEN   = "\033[32m";
    const std::string YELLOW  = "\033[33m";
    const std::string BLUE    = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN    = "\033[36m";
    const std::string WHITE   = "\033[37m";

    const std::string BG_RED     = "\033[41m";
    const std::string BG_GREEN   = "\033[42m";
    const std::string BG_YELLOW  = "\033[43m";
    const std::string BG_BLUE    = "\033[44m";
    const std::string BG_MAGENTA = "\033[45m";
    const std::string BG_CYAN    = "\033[46m";

    const std::string BRIGHT_RED     = "\033[91m";
    const std::string BRIGHT_GREEN   = "\033[92m";
    const std::string BRIGHT_YELLOW  = "\033[93m";
    const std::string BRIGHT_BLUE    = "\033[94m";
    const std::string BRIGHT_MAGENTA = "\033[95m";
    const std::string BRIGHT_CYAN    = "\033[96m";
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────
struct Config {
    double learning_rate    = 0.005;
    double momentum         = 0.9;
    double l2_lambda        = 0.0001;
    double grad_clip        = 5.0;
    double dropout_rate     = 0.3;
    int    epochs           = 300;
    int    hidden1_size     = 64;
    int    hidden2_size     = 32;
    double train_split      = 0.80;
    bool   use_bigrams      = true;
    int    early_stop_patience = 30;
    double lr_decay_factor  = 0.5;
    int    lr_decay_every   = 80;

    std::string data_file   = "data.json";
    std::string model_file  = "";
    std::string batch_file  = "";
    std::string output_file = "";
    bool   train_mode       = true;
    bool   load_model       = false;
    bool   save_model       = false;
    bool   batch_mode       = false;
    bool   verbose          = false;
    bool   show_help        = false;
};

static const char* VERSION = "2.0.0";

// ─────────────────────────────────────────────────────────────────────────────
// Data Structures
// ─────────────────────────────────────────────────────────────────────────────
struct DataPoint {
    std::string text;
    std::string label;
    std::vector<double> features;
    std::vector<double> target;
};

// ─────────────────────────────────────────────────────────────────────────────
// Utility Functions
// ─────────────────────────────────────────────────────────────────────────────
void printBanner() {
    std::cout << Color::BRIGHT_CYAN << Color::BOLD;
    std::cout << R"(
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗       ║
    ║   ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║       ║
    ║   ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║       ║
    ║   ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║       ║
    ║   ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║       ║
    ║   ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝       ║
    ║                                                           ║
    ║)" << Color::BRIGHT_YELLOW << "   Neural Network Text Classifier  v" << VERSION << Color::BRIGHT_CYAN << R"(          ║
    ║)" << Color::DIM << Color::CYAN << "   Pure C++ · TF-IDF · Multi-Layer · From Scratch" << Color::RESET << Color::BRIGHT_CYAN << Color::BOLD << R"(    ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
)" << Color::RESET;
}

void printSection(const std::string& title) {
    std::cout << "\n" << Color::BOLD << Color::BRIGHT_BLUE
              << "  ┌─ " << title << " " << std::string(std::max(0, 55 - (int)title.size()), '─') << "┐"
              << Color::RESET << "\n";
}

void printSectionEnd() {
    std::cout << Color::BOLD << Color::BRIGHT_BLUE
              << "  └" << std::string(60, '─') << "┘"
              << Color::RESET << "\n";
}

void printInfo(const std::string& key, const std::string& value) {
    std::cout << Color::DIM << "  │ " << Color::RESET
              << Color::CYAN << std::left << std::setw(28) << key << Color::RESET
              << Color::WHITE << value << Color::RESET << "\n";
}

void printInfo(const std::string& key, int value) {
    printInfo(key, std::to_string(value));
}

void printInfo(const std::string& key, double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    printInfo(key, oss.str());
}

void printSuccess(const std::string& msg) {
    std::cout << Color::BRIGHT_GREEN << "  ✓ " << Color::RESET << msg << "\n";
}

void printWarning(const std::string& msg) {
    std::cout << Color::BRIGHT_YELLOW << "  ⚠ " << Color::RESET << msg << "\n";
}

void printError(const std::string& msg) {
    std::cerr << Color::BRIGHT_RED << "  ✗ " << Color::RESET << msg << "\n";
}

std::string progressBar(double fraction, int width = 30) {
    int filled = static_cast<int>(fraction * width);
    filled = std::max(0, std::min(width, filled));
    std::string bar;
    bar += Color::BRIGHT_GREEN;
    for (int i = 0; i < filled; ++i) bar += "█";
    bar += Color::DIM;
    for (int i = filled; i < width; ++i) bar += "░";
    bar += Color::RESET;
    return bar;
}

std::string confidenceBar(double value, int width = 20) {
    int filled = static_cast<int>(value * width);
    filled = std::max(0, std::min(width, filled));
    std::string color;
    if (value >= 0.8) color = Color::BRIGHT_GREEN;
    else if (value >= 0.5) color = Color::BRIGHT_YELLOW;
    else color = Color::BRIGHT_RED;

    std::string bar;
    bar += color;
    for (int i = 0; i < filled; ++i) bar += "▓";
    bar += Color::DIM;
    for (int i = filled; i < width; ++i) bar += "░";
    bar += Color::RESET;
    return bar;
}

std::string formatDuration(double seconds) {
    std::ostringstream oss;
    if (seconds < 1.0) {
        oss << std::fixed << std::setprecision(1) << (seconds * 1000.0) << "ms";
    } else if (seconds < 60.0) {
        oss << std::fixed << std::setprecision(2) << seconds << "s";
    } else {
        int mins = static_cast<int>(seconds) / 60;
        double secs = seconds - mins * 60;
        oss << mins << "m " << std::fixed << std::setprecision(1) << secs << "s";
    }
    return oss.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON Parser (basic, handles the expected structure)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<DataPoint> parseJson(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        printError("Could not open JSON file: " + filename);
        return data;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    file.close();

    size_t arrayStart = content.find('[');
    size_t arrayEnd = content.rfind(']');
    if (arrayStart == std::string::npos || arrayEnd == std::string::npos || arrayStart >= arrayEnd) {
        printError("Could not find 'data' array in JSON.");
        return data;
    }

    std::string dataArrayStr = content.substr(arrayStart + 1, arrayEnd - arrayStart - 1);

    size_t start = dataArrayStr.find('{');
    while (start != std::string::npos) {
        size_t end = dataArrayStr.find('}', start);
        if (end == std::string::npos) break;

        std::string objectStr = dataArrayStr.substr(start + 1, end - start - 1);
        DataPoint dp;

        auto extractField = [&](const std::string& key) -> std::string {
            size_t keyPos = objectStr.find(key);
            if (keyPos == std::string::npos) return "";
            size_t colonPos = objectStr.find(':', keyPos);
            if (colonPos == std::string::npos) return "";
            size_t valStart = objectStr.find('"', colonPos);
            if (valStart == std::string::npos) return "";
            size_t valEnd = objectStr.find('"', valStart + 1);
            if (valEnd == std::string::npos) return "";
            return objectStr.substr(valStart + 1, valEnd - valStart - 1);
        };

        dp.text = extractField("text");
        dp.label = extractField("label");

        if (!dp.text.empty() && !dp.label.empty()) {
            data.push_back(dp);
        }

        start = dataArrayStr.find('{', end);
    }

    return data;
}

// ─────────────────────────────────────────────────────────────────────────────
// Text Preprocessing
// ─────────────────────────────────────────────────────────────────────────────
const std::set<std::string> STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
    "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn",
    "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
    "wasn", "weren", "won", "wouldn"
};

// Basic suffix-stripping stemmer
std::string stem(const std::string& word) {
    if (word.size() <= 3) return word;
    std::string w = word;

    // Remove common suffixes
    auto endsWith = [](const std::string& s, const std::string& suffix) {
        if (suffix.size() > s.size()) return false;
        return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    if (endsWith(w, "ingly") && w.size() > 7) w = w.substr(0, w.size() - 5);
    else if (endsWith(w, "tion") && w.size() > 6) w = w.substr(0, w.size() - 4);
    else if (endsWith(w, "ness") && w.size() > 6) w = w.substr(0, w.size() - 4);
    else if (endsWith(w, "ment") && w.size() > 6) w = w.substr(0, w.size() - 4);
    else if (endsWith(w, "able") && w.size() > 6) w = w.substr(0, w.size() - 4);
    else if (endsWith(w, "ible") && w.size() > 6) w = w.substr(0, w.size() - 4);
    else if (endsWith(w, "ally") && w.size() > 6) w = w.substr(0, w.size() - 4);
    else if (endsWith(w, "ful") && w.size() > 5) w = w.substr(0, w.size() - 3);
    else if (endsWith(w, "ous") && w.size() > 5) w = w.substr(0, w.size() - 3);
    else if (endsWith(w, "ing") && w.size() > 5) w = w.substr(0, w.size() - 3);
    else if (endsWith(w, "ied") && w.size() > 5) w = w.substr(0, w.size() - 3) + "y";
    else if (endsWith(w, "ies") && w.size() > 5) w = w.substr(0, w.size() - 3) + "y";
    else if (endsWith(w, "ed") && w.size() > 4) w = w.substr(0, w.size() - 2);
    else if (endsWith(w, "er") && w.size() > 4) w = w.substr(0, w.size() - 2);
    else if (endsWith(w, "ly") && w.size() > 4) w = w.substr(0, w.size() - 2);
    else if (endsWith(w, "es") && w.size() > 4) w = w.substr(0, w.size() - 2);
    else if (endsWith(w, "s") && !endsWith(w, "ss") && w.size() > 3) w = w.substr(0, w.size() - 1);

    return w;
}

std::vector<std::string> tokenize(const std::string& text, bool use_bigrams = false) {
    std::vector<std::string> tokens;
    std::string currentToken;

    for (char c : text) {
        if (std::isalnum(c)) {
            currentToken += std::tolower(c);
        } else if (!currentToken.empty()) {
            if (STOP_WORDS.find(currentToken) == STOP_WORDS.end()) {
                tokens.push_back(stem(currentToken));
            }
            currentToken = "";
        }
    }
    if (!currentToken.empty() && STOP_WORDS.find(currentToken) == STOP_WORDS.end()) {
        tokens.push_back(stem(currentToken));
    }

    // Add bigrams
    if (use_bigrams && tokens.size() >= 2) {
        size_t n = tokens.size();
        for (size_t i = 0; i < n - 1; ++i) {
            tokens.push_back(tokens[i] + "_" + tokens[i + 1]);
        }
    }

    return tokens;
}

std::map<std::string, int> buildVocabulary(const std::vector<DataPoint>& data, bool use_bigrams) {
    std::map<std::string, int> vocab;
    int index = 0;
    for (const auto& dp : data) {
        auto tokens = tokenize(dp.text, use_bigrams);
        for (const auto& token : tokens) {
            if (vocab.find(token) == vocab.end()) {
                vocab[token] = index++;
            }
        }
    }
    return vocab;
}

std::map<std::string, double> computeIDF(const std::vector<DataPoint>& data,
                                          const std::map<std::string, int>& vocab,
                                          bool use_bigrams) {
    int N = data.size();
    std::map<std::string, int> docFreq;

    for (const auto& dp : data) {
        auto tokens = tokenize(dp.text, use_bigrams);
        std::set<std::string> unique(tokens.begin(), tokens.end());
        for (const auto& t : unique) {
            docFreq[t]++;
        }
    }

    std::map<std::string, double> idf;
    for (const auto& [word, _] : vocab) {
        int df = 0;
        auto it = docFreq.find(word);
        if (it != docFreq.end()) df = it->second;
        idf[word] = std::log(static_cast<double>(N) / (1.0 + static_cast<double>(df)));
    }
    return idf;
}

std::vector<double> vectorize(const std::string& text,
                              const std::map<std::string, int>& vocab,
                              const std::map<std::string, double>& idf,
                              bool use_bigrams) {
    std::vector<double> features(vocab.size(), 0.0);
    auto tokens = tokenize(text, use_bigrams);
    if (tokens.empty()) return features;

    std::map<std::string, double> tf;
    for (const auto& t : tokens) tf[t]++;
    double n = static_cast<double>(tokens.size());
    for (auto& [k, v] : tf) v /= n;

    for (const auto& [token, freq] : tf) {
        auto vi = vocab.find(token);
        auto ii = idf.find(token);
        if (vi != vocab.end() && ii != idf.end()) {
            features[vi->second] = freq * ii->second;
        }
    }
    return features;
}

// ─────────────────────────────────────────────────────────────────────────────
// Neural Network — 2 Hidden Layers, Momentum SGD, Dropout, L2, Softmax
// ─────────────────────────────────────────────────────────────────────────────
class NeuralNetwork {
public:
    int inputSize, hidden1Size, hidden2Size, outputSize;

    // Weights & biases
    std::vector<std::vector<double>> w_ih1;  // input  -> hidden1
    std::vector<std::vector<double>> w_h1h2; // hidden1 -> hidden2
    std::vector<std::vector<double>> w_h2o;  // hidden2 -> output
    std::vector<double> b_h1, b_h2, b_o;

    // Momentum velocity
    std::vector<std::vector<double>> v_ih1, v_h1h2, v_h2o;
    std::vector<double> vb_h1, vb_h2, vb_o;

    // Cached forward pass
    std::vector<double> h1_raw, h1_out, h2_raw, h2_out, o_raw, o_out;
    std::vector<bool> h1_mask, h2_mask; // dropout masks

    std::mt19937 rng;

    NeuralNetwork() : inputSize(0), hidden1Size(0), hidden2Size(0), outputSize(0) {}

    NeuralNetwork(int inSize, int h1Size, int h2Size, int outSize)
        : inputSize(inSize), hidden1Size(h1Size), hidden2Size(h2Size), outputSize(outSize),
          rng(std::random_device{}())
    {
        // He initialization
        auto heInit = [&](int fan_in) {
            return std::normal_distribution<double>(0.0, std::sqrt(2.0 / fan_in));
        };

        auto initWeights = [&](std::vector<std::vector<double>>& w,
                               std::vector<std::vector<double>>& v,
                               int rows, int cols, int fan_in) {
            auto dist = heInit(fan_in);
            w.assign(rows, std::vector<double>(cols));
            v.assign(rows, std::vector<double>(cols, 0.0));
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    w[i][j] = dist(rng);
        };

        auto initBias = [](std::vector<double>& b, std::vector<double>& vb, int size) {
            b.assign(size, 0.0);
            vb.assign(size, 0.0);
        };

        initWeights(w_ih1, v_ih1, inSize, h1Size, inSize);
        initWeights(w_h1h2, v_h1h2, h1Size, h2Size, h1Size);
        initWeights(w_h2o, v_h2o, h2Size, outSize, h2Size);
        initBias(b_h1, vb_h1, h1Size);
        initBias(b_h2, vb_h2, h2Size);
        initBias(b_o, vb_o, outSize);
    }

    // ReLU
    static double relu(double x) { return std::max(0.0, x); }
    static double relu_d(double x) { return x > 0.0 ? 1.0 : 0.0; }

    // Softmax
    static std::vector<double> softmax(const std::vector<double>& z) {
        std::vector<double> out(z.size());
        double maxZ = *std::max_element(z.begin(), z.end());
        double sum = 0.0;
        for (size_t i = 0; i < z.size(); ++i) {
            out[i] = std::exp(z[i] - maxZ);
            sum += out[i];
        }
        for (auto& v : out) v /= sum;
        return out;
    }

    // Forward pass
    std::vector<double> forward(const std::vector<double>& input, bool training = false, double dropout = 0.0) {
        // Hidden layer 1
        h1_raw.resize(hidden1Size);
        h1_out.resize(hidden1Size);
        h1_mask.assign(hidden1Size, true);
        std::uniform_real_distribution<double> dropDist(0.0, 1.0);

        for (int j = 0; j < hidden1Size; ++j) {
            double sum = b_h1[j];
            for (int i = 0; i < inputSize; ++i)
                sum += input[i] * w_ih1[i][j];
            h1_raw[j] = sum;
            h1_out[j] = relu(sum);
        }

        if (training && dropout > 0.0) {
            for (int j = 0; j < hidden1Size; ++j) {
                if (dropDist(rng) < dropout) {
                    h1_mask[j] = false;
                    h1_out[j] = 0.0;
                } else {
                    h1_out[j] /= (1.0 - dropout); // inverted dropout
                }
            }
        }

        // Hidden layer 2
        h2_raw.resize(hidden2Size);
        h2_out.resize(hidden2Size);
        h2_mask.assign(hidden2Size, true);

        for (int j = 0; j < hidden2Size; ++j) {
            double sum = b_h2[j];
            for (int i = 0; i < hidden1Size; ++i)
                sum += h1_out[i] * w_h1h2[i][j];
            h2_raw[j] = sum;
            h2_out[j] = relu(sum);
        }

        if (training && dropout > 0.0) {
            for (int j = 0; j < hidden2Size; ++j) {
                if (dropDist(rng) < dropout) {
                    h2_mask[j] = false;
                    h2_out[j] = 0.0;
                } else {
                    h2_out[j] /= (1.0 - dropout);
                }
            }
        }

        // Output layer (softmax)
        o_raw.resize(outputSize);
        for (int k = 0; k < outputSize; ++k) {
            double sum = b_o[k];
            for (int j = 0; j < hidden2Size; ++j)
                sum += h2_out[j] * w_h2o[j][k];
            o_raw[k] = sum;
        }
        o_out = softmax(o_raw);
        return o_out;
    }

    // Predict (inference mode, no dropout)
    std::vector<double> predict(const std::vector<double>& input) {
        return forward(input, false, 0.0);
    }

    // Cross-entropy loss
    static double crossEntropyLoss(const std::vector<double>& pred, const std::vector<double>& target) {
        double loss = 0.0;
        for (size_t i = 0; i < pred.size(); ++i) {
            double p = std::max(pred[i], 1e-15);
            loss -= target[i] * std::log(p);
        }
        return loss;
    }

    // Backpropagation with momentum SGD + L2 + gradient clipping
    void train(const std::vector<double>& input, const std::vector<double>& target,
               double lr, double momentum, double l2, double dropout, double gradClip) {
        // Forward
        forward(input, true, dropout);

        // Output error (softmax + cross-entropy simplifies to pred - target)
        std::vector<double> d_o(outputSize);
        for (int k = 0; k < outputSize; ++k)
            d_o[k] = o_out[k] - target[k];

        // Hidden2 error
        std::vector<double> d_h2(hidden2Size, 0.0);
        for (int j = 0; j < hidden2Size; ++j) {
            if (!h2_mask[j]) continue;
            double err = 0.0;
            for (int k = 0; k < outputSize; ++k)
                err += d_o[k] * w_h2o[j][k];
            d_h2[j] = err * relu_d(h2_raw[j]);
        }

        // Hidden1 error
        std::vector<double> d_h1(hidden1Size, 0.0);
        for (int j = 0; j < hidden1Size; ++j) {
            if (!h1_mask[j]) continue;
            double err = 0.0;
            for (int k = 0; k < hidden2Size; ++k)
                err += d_h2[k] * w_h1h2[j][k];
            d_h1[j] = err * relu_d(h1_raw[j]);
        }

        // Gradient clipping helper
        auto clip = [gradClip](double g) {
            return std::max(-gradClip, std::min(gradClip, g));
        };

        // Update output weights
        for (int k = 0; k < outputSize; ++k) {
            double grad_b = clip(d_o[k]);
            vb_o[k] = momentum * vb_o[k] - lr * grad_b;
            b_o[k] += vb_o[k];
            for (int j = 0; j < hidden2Size; ++j) {
                double grad = clip(d_o[k] * h2_out[j] + l2 * w_h2o[j][k]);
                v_h2o[j][k] = momentum * v_h2o[j][k] - lr * grad;
                w_h2o[j][k] += v_h2o[j][k];
            }
        }

        // Update hidden2 weights
        for (int j = 0; j < hidden2Size; ++j) {
            double grad_b = clip(d_h2[j]);
            vb_h2[j] = momentum * vb_h2[j] - lr * grad_b;
            b_h2[j] += vb_h2[j];
            for (int i = 0; i < hidden1Size; ++i) {
                double grad = clip(d_h2[j] * h1_out[i] + l2 * w_h1h2[i][j]);
                v_h1h2[i][j] = momentum * v_h1h2[i][j] - lr * grad;
                w_h1h2[i][j] += v_h1h2[i][j];
            }
        }

        // Update hidden1 weights
        for (int j = 0; j < hidden1Size; ++j) {
            double grad_b = clip(d_h1[j]);
            vb_h1[j] = momentum * vb_h1[j] - lr * grad_b;
            b_h1[j] += vb_h1[j];
            for (int i = 0; i < inputSize; ++i) {
                if (static_cast<size_t>(i) < input.size()) {
                    double grad = clip(d_h1[j] * input[i] + l2 * w_ih1[i][j]);
                    v_ih1[i][j] = momentum * v_ih1[i][j] - lr * grad;
                    w_ih1[i][j] += v_ih1[i][j];
                }
            }
        }
    }

    // ── Model Serialization ──
    bool save(const std::string& filename,
              const std::map<std::string, int>& vocab,
              const std::map<std::string, double>& idf,
              const std::unordered_map<std::string, int>& labelToId,
              const std::vector<std::string>& idToLabel,
              bool useBigrams) const {
        std::ofstream f(filename, std::ios::binary);
        if (!f.is_open()) return false;

        // Magic + version
        const char magic[] = "SNTL";
        f.write(magic, 4);
        int version = 2;
        f.write(reinterpret_cast<const char*>(&version), sizeof(int));

        // Architecture
        f.write(reinterpret_cast<const char*>(&inputSize), sizeof(int));
        f.write(reinterpret_cast<const char*>(&hidden1Size), sizeof(int));
        f.write(reinterpret_cast<const char*>(&hidden2Size), sizeof(int));
        f.write(reinterpret_cast<const char*>(&outputSize), sizeof(int));

        // Bigrams flag
        int bg = useBigrams ? 1 : 0;
        f.write(reinterpret_cast<const char*>(&bg), sizeof(int));

        // Write weights helper
        auto writeMatrix = [&](const std::vector<std::vector<double>>& m) {
            int rows = m.size(), cols = m.empty() ? 0 : m[0].size();
            f.write(reinterpret_cast<const char*>(&rows), sizeof(int));
            f.write(reinterpret_cast<const char*>(&cols), sizeof(int));
            for (const auto& row : m)
                f.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(double));
        };
        auto writeVec = [&](const std::vector<double>& v) {
            int sz = v.size();
            f.write(reinterpret_cast<const char*>(&sz), sizeof(int));
            f.write(reinterpret_cast<const char*>(v.data()), sz * sizeof(double));
        };
        auto writeStr = [&](const std::string& s) {
            int len = s.size();
            f.write(reinterpret_cast<const char*>(&len), sizeof(int));
            f.write(s.data(), len);
        };

        writeMatrix(w_ih1);
        writeMatrix(w_h1h2);
        writeMatrix(w_h2o);
        writeVec(b_h1);
        writeVec(b_h2);
        writeVec(b_o);

        // Vocabulary
        int vocabSize = vocab.size();
        f.write(reinterpret_cast<const char*>(&vocabSize), sizeof(int));
        for (const auto& [word, idx] : vocab) {
            writeStr(word);
            f.write(reinterpret_cast<const char*>(&idx), sizeof(int));
        }

        // IDF
        int idfSize = idf.size();
        f.write(reinterpret_cast<const char*>(&idfSize), sizeof(int));
        for (const auto& [word, score] : idf) {
            writeStr(word);
            f.write(reinterpret_cast<const char*>(&score), sizeof(double));
        }

        // Labels
        int numLabels = idToLabel.size();
        f.write(reinterpret_cast<const char*>(&numLabels), sizeof(int));
        for (const auto& label : idToLabel) {
            writeStr(label);
        }

        f.close();
        return true;
    }

    bool load(const std::string& filename,
              std::map<std::string, int>& vocab,
              std::map<std::string, double>& idf,
              std::unordered_map<std::string, int>& labelToId,
              std::vector<std::string>& idToLabel,
              bool& useBigrams) {
        std::ifstream f(filename, std::ios::binary);
        if (!f.is_open()) return false;

        char magic[4];
        f.read(magic, 4);
        if (std::strncmp(magic, "SNTL", 4) != 0) {
            printError("Invalid model file format.");
            return false;
        }

        int version;
        f.read(reinterpret_cast<char*>(&version), sizeof(int));
        if (version != 2) {
            printError("Unsupported model version: " + std::to_string(version));
            return false;
        }

        f.read(reinterpret_cast<char*>(&inputSize), sizeof(int));
        f.read(reinterpret_cast<char*>(&hidden1Size), sizeof(int));
        f.read(reinterpret_cast<char*>(&hidden2Size), sizeof(int));
        f.read(reinterpret_cast<char*>(&outputSize), sizeof(int));

        int bg;
        f.read(reinterpret_cast<char*>(&bg), sizeof(int));
        useBigrams = (bg != 0);

        auto readMatrix = [&](std::vector<std::vector<double>>& m) {
            int rows, cols;
            f.read(reinterpret_cast<char*>(&rows), sizeof(int));
            f.read(reinterpret_cast<char*>(&cols), sizeof(int));
            m.assign(rows, std::vector<double>(cols));
            for (auto& row : m)
                f.read(reinterpret_cast<char*>(row.data()), cols * sizeof(double));
        };
        auto readVec = [&](std::vector<double>& v) {
            int sz;
            f.read(reinterpret_cast<char*>(&sz), sizeof(int));
            v.resize(sz);
            f.read(reinterpret_cast<char*>(v.data()), sz * sizeof(double));
        };
        auto readStr = [&]() -> std::string {
            int len;
            f.read(reinterpret_cast<char*>(&len), sizeof(int));
            std::string s(len, '\0');
            f.read(&s[0], len);
            return s;
        };

        readMatrix(w_ih1);
        readMatrix(w_h1h2);
        readMatrix(w_h2o);
        readVec(b_h1);
        readVec(b_h2);
        readVec(b_o);

        // Initialize momentum to zero
        v_ih1.assign(w_ih1.size(), std::vector<double>(w_ih1.empty() ? 0 : w_ih1[0].size(), 0.0));
        v_h1h2.assign(w_h1h2.size(), std::vector<double>(w_h1h2.empty() ? 0 : w_h1h2[0].size(), 0.0));
        v_h2o.assign(w_h2o.size(), std::vector<double>(w_h2o.empty() ? 0 : w_h2o[0].size(), 0.0));
        vb_h1.assign(b_h1.size(), 0.0);
        vb_h2.assign(b_h2.size(), 0.0);
        vb_o.assign(b_o.size(), 0.0);

        // Vocabulary
        int vocabSize;
        f.read(reinterpret_cast<char*>(&vocabSize), sizeof(int));
        vocab.clear();
        for (int i = 0; i < vocabSize; ++i) {
            std::string word = readStr();
            int idx;
            f.read(reinterpret_cast<char*>(&idx), sizeof(int));
            vocab[word] = idx;
        }

        // IDF
        int idfSize;
        f.read(reinterpret_cast<char*>(&idfSize), sizeof(int));
        idf.clear();
        for (int i = 0; i < idfSize; ++i) {
            std::string word = readStr();
            double score;
            f.read(reinterpret_cast<char*>(&score), sizeof(double));
            idf[word] = score;
        }

        // Labels
        int numLabels;
        f.read(reinterpret_cast<char*>(&numLabels), sizeof(int));
        idToLabel.resize(numLabels);
        labelToId.clear();
        for (int i = 0; i < numLabels; ++i) {
            idToLabel[i] = readStr();
            labelToId[idToLabel[i]] = i;
        }

        rng = std::mt19937(std::random_device{}());
        f.close();
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────────────
struct ClassMetrics {
    int tp = 0, fp = 0, fn = 0, tn = 0;
    double precision() const { return tp + fp > 0 ? (double)tp / (tp + fp) : 0.0; }
    double recall()    const { return tp + fn > 0 ? (double)tp / (tp + fn) : 0.0; }
    double f1()        const {
        double p = precision(), r = recall();
        return (p + r > 0) ? 2.0 * p * r / (p + r) : 0.0;
    }
};

void printConfusionMatrix(const std::vector<std::vector<int>>& cm,
                          const std::vector<std::string>& labels) {
    int n = labels.size();
    int maxLabelLen = 0;
    for (const auto& l : labels) maxLabelLen = std::max(maxLabelLen, (int)l.size());
    maxLabelLen = std::max(maxLabelLen, 10);

    std::cout << "\n" << Color::BOLD << Color::WHITE
              << "  " << std::setw(maxLabelLen + 2) << "" << "Predicted" << Color::RESET << "\n";
    std::cout << "  " << std::setw(maxLabelLen + 2) << "";
    for (const auto& l : labels)
        std::cout << Color::CYAN << std::setw(maxLabelLen) << l << Color::RESET << " ";
    std::cout << "\n";

    for (int i = 0; i < n; ++i) {
        std::cout << "  " << Color::CYAN << std::setw(maxLabelLen) << labels[i] << Color::RESET << "  ";
        for (int j = 0; j < n; ++j) {
            std::string color;
            if (i == j) color = Color::BRIGHT_GREEN;
            else if (cm[i][j] > 0) color = Color::BRIGHT_RED;
            else color = Color::DIM;
            std::cout << color << std::setw(maxLabelLen) << cm[i][j] << Color::RESET << " ";
        }
        std::cout << "\n";
    }
}

void evaluateModel(NeuralNetwork& nn,
                   const std::vector<DataPoint>& testSet,
                   const std::map<std::string, int>& vocab,
                   const std::map<std::string, double>& idf,
                   const std::unordered_map<std::string, int>& labelToId,
                   const std::vector<std::string>& idToLabel,
                   bool useBigrams) {
    int numLabels = idToLabel.size();
    std::vector<std::vector<int>> cm(numLabels, std::vector<int>(numLabels, 0));
    int correct = 0;

    for (const auto& dp : testSet) {
        auto features = vectorize(dp.text, vocab, idf, useBigrams);
        auto pred = nn.predict(features);
        int predId = std::distance(pred.begin(), std::max_element(pred.begin(), pred.end()));
        int trueId = labelToId.at(dp.label);
        cm[trueId][predId]++;
        if (predId == trueId) correct++;
    }

    double accuracy = testSet.empty() ? 0.0 : (double)correct / testSet.size();

    printSection("Evaluation Results");
    printInfo("Test samples", (int)testSet.size());
    printInfo("Overall accuracy", std::to_string((int)(accuracy * 100)) + "%");

    // Per-class metrics
    std::cout << "\n" << Color::BOLD << "  Per-Class Metrics:" << Color::RESET << "\n";
    std::cout << Color::DIM << "  "
              << std::setw(12) << "Class"
              << std::setw(12) << "Precision"
              << std::setw(12) << "Recall"
              << std::setw(12) << "F1-Score"
              << std::setw(10) << "Support"
              << Color::RESET << "\n";
    std::cout << Color::DIM << "  " << std::string(58, '─') << Color::RESET << "\n";

    double macroF1 = 0.0;
    for (int c = 0; c < numLabels; ++c) {
        ClassMetrics m;
        for (int i = 0; i < numLabels; ++i) {
            for (int j = 0; j < numLabels; ++j) {
                if (i == c && j == c) m.tp += cm[i][j];
                else if (i != c && j == c) m.fp += cm[i][j];
                else if (i == c && j != c) m.fn += cm[i][j];
                else m.tn += cm[i][j];
            }
        }
        int support = m.tp + m.fn;
        macroF1 += m.f1();

        std::string labelColor;
        if (idToLabel[c] == "positive") labelColor = Color::BRIGHT_GREEN;
        else if (idToLabel[c] == "negative") labelColor = Color::BRIGHT_RED;
        else labelColor = Color::BRIGHT_YELLOW;

        std::cout << "  " << labelColor << std::setw(12) << idToLabel[c] << Color::RESET;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::setw(12) << m.precision()
                  << std::setw(12) << m.recall()
                  << std::setw(12) << m.f1()
                  << std::setw(10) << support << "\n";
    }
    macroF1 /= numLabels;
    std::cout << Color::DIM << "  " << std::string(58, '─') << Color::RESET << "\n";
    std::cout << "  " << Color::BOLD << std::setw(12) << "Macro F1" << Color::RESET
              << std::fixed << std::setprecision(3) << std::setw(36) << macroF1 << "\n";

    // Confusion matrix
    std::cout << "\n" << Color::BOLD << "  Confusion Matrix:" << Color::RESET << "\n";
    printConfusionMatrix(cm, idToLabel);
    printSectionEnd();
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature Importance for a single prediction
// ─────────────────────────────────────────────────────────────────────────────
void showFeatureImportance(const std::string& text,
                           const std::vector<double>& features,
                           const std::map<std::string, int>& vocab,
                           int predictedClass,
                           const NeuralNetwork& nn,
                           int topN = 5) {
    // Approximate importance: feature_value * sum of absolute weights to predicted class
    std::vector<std::pair<std::string, double>> importance;

    for (const auto& [word, idx] : vocab) {
        if (features[idx] == 0.0) continue;
        double score = 0.0;
        // Trace through both hidden layers (approximate)
        for (int h1 = 0; h1 < nn.hidden1Size; ++h1) {
            double w1 = nn.w_ih1[idx][h1];
            for (int h2 = 0; h2 < nn.hidden2Size; ++h2) {
                double w2 = nn.w_h1h2[h1][h2];
                double w3 = nn.w_h2o[h2][predictedClass];
                score += features[idx] * w1 * w2 * w3;
            }
        }
        importance.push_back({word, score});
    }

    std::sort(importance.begin(), importance.end(),
              [](const auto& a, const auto& b) { return std::abs(a.second) > std::abs(b.second); });

    int shown = std::min(topN, (int)importance.size());
    if (shown > 0) {
        std::cout << Color::DIM << "  │ " << Color::RESET
                  << Color::BOLD << "Key contributing words:" << Color::RESET << "\n";
        for (int i = 0; i < shown; ++i) {
            std::string arrow = importance[i].second > 0 ? "↑" : "↓";
            std::string color = importance[i].second > 0 ? Color::BRIGHT_GREEN : Color::BRIGHT_RED;
            std::cout << Color::DIM << "  │   " << Color::RESET
                      << color << arrow << " " << importance[i].first << Color::RESET
                      << Color::DIM << " (" << std::fixed << std::setprecision(4)
                      << importance[i].second << ")" << Color::RESET << "\n";
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI Argument Parsing
// ─────────────────────────────────────────────────────────────────────────────
void printHelp(const char* progName) {
    std::cout << Color::BOLD << "\nUsage: " << Color::RESET << progName << " [OPTIONS] <data.json>\n\n";
    std::cout << Color::BOLD << "Options:" << Color::RESET << "\n";
    std::cout << "  " << Color::CYAN << "--help, -h" << Color::RESET
              << "              Show this help message\n";
    std::cout << "  " << Color::CYAN << "--version, -v" << Color::RESET
              << "           Show version information\n";
    std::cout << "  " << Color::CYAN << "--train" << Color::RESET
              << "                 Train a new model (default)\n";
    std::cout << "  " << Color::CYAN << "--save <file>" << Color::RESET
              << "           Save trained model to file\n";
    std::cout << "  " << Color::CYAN << "--load <file>" << Color::RESET
              << "           Load model from file (skip training)\n";
    std::cout << "  " << Color::CYAN << "--batch <file>" << Color::RESET
              << "          Classify texts from file (one per line)\n";
    std::cout << "  " << Color::CYAN << "--output <file>" << Color::RESET
              << "         Output file for batch results (default: stdout)\n";
    std::cout << "  " << Color::CYAN << "--epochs <n>" << Color::RESET
              << "            Number of training epochs (default: 300)\n";
    std::cout << "  " << Color::CYAN << "--lr <rate>" << Color::RESET
              << "             Learning rate (default: 0.005)\n";
    std::cout << "  " << Color::CYAN << "--hidden1 <n>" << Color::RESET
              << "           Hidden layer 1 size (default: 64)\n";
    std::cout << "  " << Color::CYAN << "--hidden2 <n>" << Color::RESET
              << "           Hidden layer 2 size (default: 32)\n";
    std::cout << "  " << Color::CYAN << "--no-bigrams" << Color::RESET
              << "            Disable bigram features\n";
    std::cout << "  " << Color::CYAN << "--verbose" << Color::RESET
              << "               Show detailed training output\n";
    std::cout << "\n" << Color::BOLD << "Examples:" << Color::RESET << "\n";
    std::cout << Color::DIM << "  # Train and enter interactive mode\n" << Color::RESET;
    std::cout << "  " << progName << " data.json\n\n";
    std::cout << Color::DIM << "  # Train and save model\n" << Color::RESET;
    std::cout << "  " << progName << " data.json --save model.sntl\n\n";
    std::cout << Color::DIM << "  # Load model and classify a batch\n" << Color::RESET;
    std::cout << "  " << progName << " data.json --load model.sntl --batch texts.txt\n\n";
}

Config parseArgs(int argc, char* argv[]) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") { cfg.show_help = true; return cfg; }
        if (arg == "--version" || arg == "-v") {
            std::cout << "SentinelNN v" << VERSION << "\n";
            std::exit(0);
        }
        if (arg == "--train") { cfg.train_mode = true; continue; }
        if (arg == "--verbose") { cfg.verbose = true; continue; }
        if (arg == "--no-bigrams") { cfg.use_bigrams = false; continue; }

        if (arg == "--save" && i + 1 < argc) { cfg.save_model = true; cfg.model_file = argv[++i]; continue; }
        if (arg == "--load" && i + 1 < argc) { cfg.load_model = true; cfg.model_file = argv[++i]; cfg.train_mode = false; continue; }
        if (arg == "--batch" && i + 1 < argc) { cfg.batch_mode = true; cfg.batch_file = argv[++i]; continue; }
        if (arg == "--output" && i + 1 < argc) { cfg.output_file = argv[++i]; continue; }
        if (arg == "--epochs" && i + 1 < argc) { cfg.epochs = std::stoi(argv[++i]); continue; }
        if (arg == "--lr" && i + 1 < argc) { cfg.learning_rate = std::stod(argv[++i]); continue; }
        if (arg == "--hidden1" && i + 1 < argc) { cfg.hidden1_size = std::stoi(argv[++i]); continue; }
        if (arg == "--hidden2" && i + 1 < argc) { cfg.hidden2_size = std::stoi(argv[++i]); continue; }

        // Positional: data file
        if (arg[0] != '-') {
            cfg.data_file = arg;
        } else {
            printWarning("Unknown option: " + arg);
        }
    }

    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Config cfg = parseArgs(argc, argv);

    if (cfg.show_help) {
        printBanner();
        printHelp(argv[0]);
        return 0;
    }

    printBanner();

    // ── Shared state ──
    std::map<std::string, int> vocabulary;
    std::map<std::string, double> idfScores;
    std::unordered_map<std::string, int> labelToId;
    std::vector<std::string> idToLabel;
    NeuralNetwork nn;
    bool useBigrams = cfg.use_bigrams;

    // ── Load Model ──
    if (cfg.load_model) {
        printSection("Loading Model");
        printInfo("Model file", cfg.model_file);

        if (nn.load(cfg.model_file, vocabulary, idfScores, labelToId, idToLabel, useBigrams)) {
            printSuccess("Model loaded successfully!");
            printInfo("Architecture", std::to_string(nn.inputSize) + " → " +
                      std::to_string(nn.hidden1Size) + " → " +
                      std::to_string(nn.hidden2Size) + " → " +
                      std::to_string(nn.outputSize));
            printInfo("Vocabulary size", (int)vocabulary.size());
            printInfo("Labels", std::to_string(idToLabel.size()));
            printInfo("Bigrams", useBigrams ? "enabled" : "disabled");
            printSectionEnd();
        } else {
            printError("Failed to load model from: " + cfg.model_file);
            return 1;
        }
    }

    // ── Train Mode ──
    if (cfg.train_mode) {
        auto totalStart = std::chrono::high_resolution_clock::now();

        // Load data
        printSection("Loading Data");
        printInfo("Data file", cfg.data_file);
        std::vector<DataPoint> dataset = parseJson(cfg.data_file);
        if (dataset.empty()) {
            printError("No data loaded. Exiting.");
            return 1;
        }
        printSuccess("Loaded " + std::to_string(dataset.size()) + " samples");

        // Count per label
        std::map<std::string, int> labelCounts;
        for (const auto& dp : dataset) labelCounts[dp.label]++;
        for (const auto& [label, count] : labelCounts) {
            std::string color;
            if (label == "positive") color = Color::BRIGHT_GREEN;
            else if (label == "negative") color = Color::BRIGHT_RED;
            else color = Color::BRIGHT_YELLOW;
            std::cout << Color::DIM << "  │ " << Color::RESET
                      << color << "  " << label << Color::RESET
                      << ": " << count << " samples\n";
        }
        printSectionEnd();

        // Build vocabulary & features
        printSection("Feature Engineering");
        vocabulary = buildVocabulary(dataset, useBigrams);
        printInfo("Vocabulary size", (int)vocabulary.size());
        printInfo("Bigrams", useBigrams ? "enabled" : "disabled");

        idfScores = computeIDF(dataset, vocabulary, useBigrams);

        // Assign labels
        int labelCounter = 0;
        labelToId.clear();
        for (auto& dp : dataset) {
            if (labelToId.find(dp.label) == labelToId.end()) {
                labelToId[dp.label] = labelCounter++;
            }
        }
        int numLabels = labelCounter;
        idToLabel.resize(numLabels);
        for (const auto& [label, id] : labelToId) idToLabel[id] = label;

        printInfo("Classes", numLabels);

        // Vectorize
        for (auto& dp : dataset) {
            dp.features = vectorize(dp.text, vocabulary, idfScores, useBigrams);
            dp.target.assign(numLabels, 0.0);
            dp.target[labelToId[dp.label]] = 1.0;
        }
        printSuccess("TF-IDF vectorization complete");
        printSectionEnd();

        // Train/test split (stratified)
        printSection("Data Split");
        std::mt19937 splitRng(42);
        std::shuffle(dataset.begin(), dataset.end(), splitRng);

        // Group by label for stratified split
        std::map<std::string, std::vector<DataPoint>> byLabel;
        for (const auto& dp : dataset) byLabel[dp.label].push_back(dp);

        std::vector<DataPoint> trainSet, testSet;
        for (auto& [label, points] : byLabel) {
            std::shuffle(points.begin(), points.end(), splitRng);
            int trainCount = static_cast<int>(points.size() * cfg.train_split);
            for (int i = 0; i < (int)points.size(); ++i) {
                if (i < trainCount) trainSet.push_back(points[i]);
                else testSet.push_back(points[i]);
            }
        }

        printInfo("Training samples", (int)trainSet.size());
        printInfo("Test samples", (int)testSet.size());
        printInfo("Split ratio", std::to_string((int)(cfg.train_split * 100)) + "/" +
                  std::to_string((int)((1.0 - cfg.train_split) * 100)));
        printSectionEnd();

        // Initialize network
        int inputSize = vocabulary.size();
        int outputSize = numLabels;
        nn = NeuralNetwork(inputSize, cfg.hidden1_size, cfg.hidden2_size, outputSize);

        printSection("Training");
        printInfo("Architecture", std::to_string(inputSize) + " → " +
                  std::to_string(cfg.hidden1_size) + " → " +
                  std::to_string(cfg.hidden2_size) + " → " +
                  std::to_string(outputSize));
        printInfo("Learning rate", cfg.learning_rate);
        printInfo("Momentum", cfg.momentum);
        printInfo("L2 regularization", cfg.l2_lambda);
        printInfo("Dropout", cfg.dropout_rate);
        printInfo("Gradient clip", cfg.grad_clip);
        printInfo("Epochs", cfg.epochs);
        printInfo("LR decay", "×" + std::to_string(cfg.lr_decay_factor) + " every " +
                  std::to_string(cfg.lr_decay_every) + " epochs");
        printInfo("Early stopping", std::to_string(cfg.early_stop_patience) + " epochs patience");
        std::cout << "\n";

        std::mt19937 trainRng(std::random_device{}());
        double bestLoss = std::numeric_limits<double>::max();
        int patienceCounter = 0;
        double currentLR = cfg.learning_rate;

        auto trainStart = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
            std::shuffle(trainSet.begin(), trainSet.end(), trainRng);

            // LR decay
            if (epoch > 0 && epoch % cfg.lr_decay_every == 0) {
                currentLR *= cfg.lr_decay_factor;
            }

            double epochLoss = 0.0;
            for (const auto& dp : trainSet) {
                nn.train(dp.features, dp.target, currentLR, cfg.momentum,
                         cfg.l2_lambda, cfg.dropout_rate, cfg.grad_clip);
                auto pred = nn.predict(dp.features);
                epochLoss += NeuralNetwork::crossEntropyLoss(pred, dp.target);
            }
            double avgLoss = epochLoss / trainSet.size();

            // Compute train accuracy
            int trainCorrect = 0;
            for (const auto& dp : trainSet) {
                auto pred = nn.predict(dp.features);
                int predId = std::distance(pred.begin(), std::max_element(pred.begin(), pred.end()));
                int trueId = labelToId[dp.label];
                if (predId == trueId) trainCorrect++;
            }
            double trainAcc = (double)trainCorrect / trainSet.size();

            // Early stopping check
            if (avgLoss < bestLoss - 1e-6) {
                bestLoss = avgLoss;
                patienceCounter = 0;
            } else {
                patienceCounter++;
            }

            // Progress display
            double frac = (double)(epoch + 1) / cfg.epochs;
            std::cout << "\r  " << progressBar(frac, 25)
                      << Color::DIM << " Epoch " << Color::RESET
                      << Color::BOLD << std::setw(4) << (epoch + 1) << Color::RESET
                      << Color::DIM << "/" << cfg.epochs << Color::RESET
                      << "  Loss: " << Color::BRIGHT_YELLOW << std::fixed << std::setprecision(4) << avgLoss << Color::RESET
                      << "  Acc: " << Color::BRIGHT_GREEN << std::fixed << std::setprecision(1) << (trainAcc * 100) << "%" << Color::RESET
                      << "  LR: " << Color::DIM << std::scientific << std::setprecision(1) << currentLR << Color::RESET
                      << "    " << std::flush;

            if (cfg.verbose && (epoch + 1) % 50 == 0) {
                std::cout << "\n";
            }

            if (patienceCounter >= cfg.early_stop_patience) {
                std::cout << "\n";
                printWarning("Early stopping triggered at epoch " + std::to_string(epoch + 1) +
                             " (no improvement for " + std::to_string(cfg.early_stop_patience) + " epochs)");
                break;
            }
        }

        auto trainEnd = std::chrono::high_resolution_clock::now();
        double trainTime = std::chrono::duration<double>(trainEnd - trainStart).count();

        std::cout << "\n";
        printSuccess("Training complete in " + formatDuration(trainTime));
        printInfo("Best loss achieved", bestLoss);
        printSectionEnd();

        // Evaluate on test set
        evaluateModel(nn, testSet, vocabulary, idfScores, labelToId, idToLabel, useBigrams);

        // Save model
        if (cfg.save_model) {
            printSection("Saving Model");
            printInfo("Output file", cfg.model_file);
            if (nn.save(cfg.model_file, vocabulary, idfScores, labelToId, idToLabel, useBigrams)) {
                printSuccess("Model saved successfully!");
            } else {
                printError("Failed to save model.");
            }
            printSectionEnd();
        }

        auto totalEnd = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double>(totalEnd - totalStart).count();
        std::cout << Color::DIM << "\n  Total pipeline time: " << formatDuration(totalTime) << Color::RESET << "\n";
    }

    // ── Batch Mode ──
    if (cfg.batch_mode) {
        printSection("Batch Classification");
        printInfo("Input file", cfg.batch_file);

        std::ifstream batchIn(cfg.batch_file);
        if (!batchIn.is_open()) {
            printError("Could not open batch file: " + cfg.batch_file);
            return 1;
        }

        std::ofstream outFile;
        std::ostream* out = &std::cout;
        if (!cfg.output_file.empty()) {
            outFile.open(cfg.output_file);
            if (!outFile.is_open()) {
                printError("Could not open output file: " + cfg.output_file);
                return 1;
            }
            out = &outFile;
            printInfo("Output file", cfg.output_file);
        }

        std::string line;
        int count = 0;
        while (std::getline(batchIn, line)) {
            if (line.empty()) continue;
            auto features = vectorize(line, vocabulary, idfScores, useBigrams);
            auto pred = nn.predict(features);
            int predId = std::distance(pred.begin(), std::max_element(pred.begin(), pred.end()));
            double conf = pred[predId];

            *out << idToLabel[predId] << "\t"
                 << std::fixed << std::setprecision(4) << conf << "\t"
                 << line << "\n";
            count++;
        }

        printSuccess("Classified " + std::to_string(count) + " texts");
        printSectionEnd();

        if (!cfg.output_file.empty()) {
            outFile.close();
        }
        return 0;
    }

    // ── Interactive Mode ──
    printSection("Interactive Classification");
    std::cout << Color::DIM << "  │ " << Color::RESET
              << "Enter text to classify. Commands:\n";
    std::cout << Color::DIM << "  │ " << Color::RESET
              << Color::CYAN << "  quit" << Color::RESET << "     Exit the program\n";
    std::cout << Color::DIM << "  │ " << Color::RESET
              << Color::CYAN << "  stats" << Color::RESET << "    Show session statistics\n";
    std::cout << Color::DIM << "  │ " << Color::RESET
              << Color::CYAN << "  clear" << Color::RESET << "    Clear screen\n";
    printSectionEnd();

    int sessionTotal = 0;
    std::map<std::string, int> sessionCounts;
    std::string inputText;

    while (true) {
        std::cout << "\n" << Color::BRIGHT_CYAN << "  ❯ " << Color::RESET;
        if (!std::getline(std::cin, inputText)) {
            if (std::cin.eof()) {
                std::cout << "\n";
                break;
            }
            break;
        }

        // Trim
        size_t start = inputText.find_first_not_of(" \t\r\n");
        size_t end = inputText.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        inputText = inputText.substr(start, end - start + 1);

        if (inputText.empty()) continue;

        if (inputText == "quit" || inputText == "exit" || inputText == "q") break;

        if (inputText == "clear") {
            std::cout << "\033[2J\033[H";
            continue;
        }

        if (inputText == "stats") {
            printSection("Session Statistics");
            printInfo("Total classifications", sessionTotal);
            for (const auto& [label, count] : sessionCounts) {
                std::string color;
                if (label == "positive") color = Color::BRIGHT_GREEN;
                else if (label == "negative") color = Color::BRIGHT_RED;
                else color = Color::BRIGHT_YELLOW;
                std::cout << Color::DIM << "  │ " << Color::RESET
                          << color << "  " << label << Color::RESET
                          << ": " << count << "\n";
            }
            printSectionEnd();
            continue;
        }

        // Classify
        auto features = vectorize(inputText, vocabulary, idfScores, useBigrams);
        auto predictions = nn.predict(features);

        int predId = std::distance(predictions.begin(),
                                   std::max_element(predictions.begin(), predictions.end()));
        std::string predLabel = idToLabel[predId];
        double confidence = predictions[predId];

        sessionTotal++;
        sessionCounts[predLabel]++;

        // Display result
        std::string labelColor;
        std::string emoji;
        if (predLabel == "positive") { labelColor = Color::BRIGHT_GREEN; emoji = "😊"; }
        else if (predLabel == "negative") { labelColor = Color::BRIGHT_RED; emoji = "😞"; }
        else { labelColor = Color::BRIGHT_YELLOW; emoji = "😐"; }

        std::cout << "\n"
                  << Color::DIM << "  ┌─ Result ──────────────────────────────────────────┐" << Color::RESET << "\n"
                  << Color::DIM << "  │ " << Color::RESET
                  << "Sentiment:  " << labelColor << Color::BOLD << predLabel
                  << Color::RESET << "  " << emoji << "\n"
                  << Color::DIM << "  │ " << Color::RESET
                  << "Confidence: " << confidenceBar(confidence) << " "
                  << Color::BOLD << std::fixed << std::setprecision(1) << (confidence * 100) << "%" << Color::RESET << "\n";

        // Show all class probabilities
        std::cout << Color::DIM << "  │ " << Color::RESET
                  << Color::BOLD << "Class probabilities:" << Color::RESET << "\n";
        for (int c = 0; c < (int)idToLabel.size(); ++c) {
            std::string cColor;
            if (idToLabel[c] == "positive") cColor = Color::BRIGHT_GREEN;
            else if (idToLabel[c] == "negative") cColor = Color::BRIGHT_RED;
            else cColor = Color::BRIGHT_YELLOW;

            std::cout << Color::DIM << "  │   " << Color::RESET
                      << cColor << std::setw(10) << idToLabel[c] << Color::RESET
                      << "  " << confidenceBar(predictions[c], 15) << " "
                      << std::fixed << std::setprecision(1) << (predictions[c] * 100) << "%\n";
        }

        // Feature importance
        showFeatureImportance(inputText, features, vocabulary, predId, nn, 5);

        std::cout << Color::DIM << "  └────────────────────────────────────────────────────┘" << Color::RESET << "\n";
    }

    // Session summary
    if (sessionTotal > 0) {
        std::cout << "\n";
        printSection("Session Summary");
        printInfo("Total classifications", sessionTotal);
        for (const auto& [label, count] : sessionCounts) {
            double pct = (double)count / sessionTotal * 100;
            printInfo("  " + label, std::to_string(count) + " (" + std::to_string((int)pct) + "%)");
        }
        printSectionEnd();
    }

    std::cout << "\n" << Color::DIM << "  Thank you for using SentinelNN! 👋" << Color::RESET << "\n\n";
    return 0;
}
