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
#include <limits> // Required for numeric_limits
#include <map>    // Use map for ordered vocabulary iteration
#include <set>    // For stop words



      /*
       O
      /|\
      / \
     {}  {}
Yunus is an organic machine */



// --- Configuration ---
const double LEARNING_RATE = 0.01; // Reverted learning rate
const int EPOCHS = 500; // Increased epochs significantly
const int HIDDEN_SIZE = 20; // Increased hidden layer size

// --- Data Structures ---
struct DataPoint {
    std::string text;
    std::string label;
    std::vector<double> features;
    std::vector<double> target; // One-hot encoded label
};

// --- Basic JSON Parser ---
// WARNING: Very basic parser, assumes specific structure and no nested objects/arrays beyond the top level.
// Only handles {"data": [{"text": "...", "label": "..."}, ...]}
std::vector<DataPoint> parseJson(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open JSON file: " << filename << std::endl;
        return data;
    }

    std::string line;
    std::string content;
    while (std::getline(file, line)) {
        content += line;
    }
    file.close();

    // Find the start and end of the main "data" array
    size_t arrayStart = content.find('[');
    size_t arrayEnd = content.rfind(']');
    if (arrayStart == std::string::npos || arrayEnd == std::string::npos || arrayStart >= arrayEnd) {
        std::cerr << "Error: Could not find 'data' array in JSON." << std::endl;
        return data;
    }

    std::string dataArrayStr = content.substr(arrayStart + 1, arrayEnd - arrayStart - 1);

    // Split into objects (crude split by '}')
    size_t start = dataArrayStr.find('{');
    while (start != std::string::npos) {
        size_t end = dataArrayStr.find('}', start);
        if (end == std::string::npos) break;

        std::string objectStr = dataArrayStr.substr(start + 1, end - start - 1);
        DataPoint dp;

        // Extract text
        size_t textKeyPos = objectStr.find("text");
        if (textKeyPos != std::string::npos) {
            size_t textStart = objectStr.find(':', textKeyPos);
            size_t textValueStart = objectStr.find('"', textStart);
            size_t textValueEnd = objectStr.find('"', textValueStart + 1);
            if (textStart != std::string::npos && textValueStart != std::string::npos && textValueEnd != std::string::npos) {
                dp.text = objectStr.substr(textValueStart + 1, textValueEnd - textValueStart - 1);
            } else {
                 std::cerr << "Warning: Could not parse 'text' in object: " << objectStr << std::endl;
            }
        } else {
             std::cerr << "Warning: Could not find 'text' key in object: " << objectStr << std::endl;
        }


        // Extract label
         size_t labelKeyPos = objectStr.find("label");
        if (labelKeyPos != std::string::npos) {
            size_t labelStart = objectStr.find(':', labelKeyPos);
            size_t labelValueStart = objectStr.find('"', labelStart);
            size_t labelValueEnd = objectStr.find('"', labelValueStart + 1);
             if (labelStart != std::string::npos && labelValueStart != std::string::npos && labelValueEnd != std::string::npos) {
                dp.label = objectStr.substr(labelValueStart + 1, labelValueEnd - labelValueStart - 1);
            } else {
                 std::cerr << "Warning: Could not parse 'label' in object: " << objectStr << std::endl;
            }
        } else {
             std::cerr << "Warning: Could not find 'label' key in object: " << objectStr << std::endl;
        }


        if (!dp.text.empty() && !dp.label.empty()) {
           data.push_back(dp);
        }

        start = dataArrayStr.find('{', end);
    }


    if (data.empty()) {
         std::cerr << "Error: No valid data points parsed from JSON." << std::endl;
    }


    return data;
}


// --- Text Preprocessing ---

// Basic English stop words list
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
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
};

std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string currentToken;
    for (char c : text) {
        if (std::isalnum(c)) {
            currentToken += std::tolower(c);
        } else if (!currentToken.empty()) {
            // Only add token if it's not a stop word
            if (STOP_WORDS.find(currentToken) == STOP_WORDS.end()) {
                 tokens.push_back(currentToken);
            }
            currentToken = "";
        }
    }
    if (!currentToken.empty() && STOP_WORDS.find(currentToken) == STOP_WORDS.end()) {
        tokens.push_back(currentToken);
    }
    return tokens;
}

// Creates a vocabulary and maps words to indices
std::map<std::string, int> buildVocabulary(const std::vector<DataPoint>& data) {
    std::map<std::string, int> vocab; // Use map for consistent ordering
    int index = 0;
    for (const auto& dp : data) {
        std::vector<std::string> tokens = tokenize(dp.text);
        for (const std::string& token : tokens) {
            if (vocab.find(token) == vocab.end()) {
                vocab[token] = index++;
            }
        }
    }
    return vocab;
}

// Creates TF-IDF features
std::vector<double> vectorize(const std::string& text,
                              const std::map<std::string, int>& vocab,
                              const std::map<std::string, double>& idfScores) {
    std::vector<double> features(vocab.size(), 0.0);
    std::vector<std::string> tokens = tokenize(text);

    if (tokens.empty()) {
        return features; // Return zero vector if no valid tokens
    }

    // Calculate Term Frequency (TF) for the current text
    std::map<std::string, double> termFrequency;
    for (const std::string& token : tokens) {
        termFrequency[token]++;
    }
    // Normalize TF by the number of tokens in the document
    double numTokens = static_cast<double>(tokens.size());
    for (auto& pair : termFrequency) {
        pair.second /= numTokens;
    }

    // Calculate TF-IDF score for each word in the vocabulary
    for (const auto& pair : termFrequency) {
        const std::string& token = pair.first;
        double tf = pair.second;

        auto vocab_it = vocab.find(token);
        auto idf_it = idfScores.find(token);

        if (vocab_it != vocab.end() && idf_it != idfScores.end()) {
            int vocabIndex = vocab_it->second;
            double idf = idf_it->second;
            features[vocabIndex] = tf * idf;
        }
    }
    return features;
}

// --- Neural Network Components ---
// Activation function (Sigmoid)
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Derivative of Sigmoid
double sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

// Activation function (ReLU)
double relu(double x) {
    return std::max(0.0, x);
}

// Derivative of ReLU
double relu_derivative(double x) {
    return (x > 0.0) ? 1.0 : 0.0;
}

// --- Neural Network Class ---
class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        : inputSize_(inputSize), hiddenSize_(hiddenSize), outputSize_(outputSize) {
        // Initialize weights and biases randomly
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(-0.5, 0.5);

        // Weights between input and hidden layer
        weights_ih_.resize(inputSize_, std::vector<double>(hiddenSize_));
        for (int i = 0; i < inputSize_; ++i) {
            for (int j = 0; j < hiddenSize_; ++j) {
                weights_ih_[i][j] = dist(rng);
            }
        }

        // Biases for hidden layer
        bias_h_.resize(hiddenSize_);
        for (int j = 0; j < hiddenSize_; ++j) {
            bias_h_[j] = dist(rng);
        }

        // Weights between hidden and output layer
        weights_ho_.resize(hiddenSize_, std::vector<double>(outputSize_));
        for (int i = 0; i < hiddenSize_; ++i) {
            for (int j = 0; j < outputSize_; ++j) {
                weights_ho_[i][j] = dist(rng);
            }
        }

        // Biases for output layer
        bias_o_.resize(outputSize_);
        for (int j = 0; j < outputSize_; ++j) {
            bias_o_[j] = dist(rng);
        }
    }

    // Forward pass
    std::vector<double> predict(const std::vector<double>& inputs) {
        // Calculate hidden layer activations
        hidden_outputs_.resize(hiddenSize_);
        hidden_raw_.resize(hiddenSize_); // Store pre-activation values for backprop
        for (int j = 0; j < hiddenSize_; ++j) {
            double sum = bias_h_[j];
            for (int i = 0; i < inputSize_; ++i) {
                sum += inputs[i] * weights_ih_[i][j];
            }
             hidden_raw_[j] = sum;
            hidden_outputs_[j] = relu(sum); // Use ReLU for hidden layer
        }

        // Calculate output layer activations
        std::vector<double> final_outputs(outputSize_);
        output_raw_.resize(outputSize_); // Store pre-activation values for backprop
        for (int k = 0; k < outputSize_; ++k) {
            double sum = bias_o_[k];
            for (int j = 0; j < hiddenSize_; ++j) {
                sum += hidden_outputs_[j] * weights_ho_[j][k];
            }
             output_raw_[k] = sum;
            final_outputs[k] = sigmoid(sum); // Using sigmoid for output as well (simple case)
        }
        return final_outputs;
    }

    // Backpropagation and weight update
    void train(const std::vector<double>& inputs, const std::vector<double>& targets) {
        // 1. Forward pass (already done if predict was called just before)
        //    We reuse hidden_outputs_, hidden_raw_, output_raw_ from the last forward pass
         std::vector<double> outputs = predict(inputs); // Ensure forward pass happens & caches values

        // 2. Calculate output layer errors (delta_k)
        std::vector<double> output_errors(outputSize_);
        for (int k = 0; k < outputSize_; ++k) {
            output_errors[k] = (targets[k] - outputs[k]) * sigmoid_derivative(output_raw_[k]);
        }

        // 3. Calculate hidden layer errors (delta_j)
        std::vector<double> hidden_errors(hiddenSize_);
        for (int j = 0; j < hiddenSize_; ++j) {
            double error_sum = 0.0;
            for (int k = 0; k < outputSize_; ++k) {
                error_sum += output_errors[k] * weights_ho_[j][k];
            }
            hidden_errors[j] = error_sum * relu_derivative(hidden_raw_[j]); // Use ReLU derivative
        }

        // 4. Update output layer weights and biases
        for (int k = 0; k < outputSize_; ++k) {
            bias_o_[k] += LEARNING_RATE * output_errors[k];
            for (int j = 0; j < hiddenSize_; ++j) {
                weights_ho_[j][k] += LEARNING_RATE * output_errors[k] * hidden_outputs_[j];
            }
        }

        // 5. Update hidden layer weights and biases
        for (int j = 0; j < hiddenSize_; ++j) {
            bias_h_[j] += LEARNING_RATE * hidden_errors[j];
            for (int i = 0; i < inputSize_; ++i) {
                 // Ensure 'inputs' is valid and sized correctly (inputSize_)
                 // Using size_t for comparison with vector size
                 if (static_cast<size_t>(i) < inputs.size()) { // Explicit cast to size_t for comparison
                      weights_ih_[i][j] += LEARNING_RATE * hidden_errors[j] * inputs[i];
                 } else {
                      std::cerr << "Warning: Input index out of bounds during weight update (i=" << i << ", inputs.size()=" << inputs.size() << ")" << std::endl;
                 }
            }
        }
    }

    // Calculate Mean Squared Error for a single prediction
    double calculate_mse(const std::vector<double>& predictions, const std::vector<double>& targets) {
        if (predictions.size() != targets.size() || predictions.empty()) {
            return 0.0; // Or handle error appropriately
        }
        double mse = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            mse += std::pow(targets[i] - predictions[i], 2);
        }
        return mse / predictions.size();
    }

private:
    int inputSize_;
    int hiddenSize_;
    int outputSize_;

    // Weights
    std::vector<std::vector<double>> weights_ih_; // Input -> Hidden
    std::vector<std::vector<double>> weights_ho_; // Hidden -> Output

    // Biases
    std::vector<double> bias_h_; // Hidden layer
    std::vector<double> bias_o_; // Output layer

     // Cached values from forward pass for backpropagation
    std::vector<double> hidden_outputs_;
    std::vector<double> hidden_raw_; // Pre-activation hidden values
    std::vector<double> output_raw_; // Pre-activation output values
};

// --- Main Application Logic ---
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_data.json>" << std::endl;
        return 1;
    }
    std::string json_filename = argv[1];

    // 1. Load and Parse Data
    std::cout << "Loading data from " << json_filename << "..." << std::endl;
    std::vector<DataPoint> dataset = parseJson(json_filename);
    if (dataset.empty()) {
        std::cerr << "Failed to load or parse data. Exiting." << std::endl;
        return 1;
    }
    std::cout << "Loaded " << dataset.size() << " data points." << std::endl;


    // 2. Preprocess Data
    std::cout << "Building vocabulary..." << std::endl;
    std::map<std::string, int> vocabulary = buildVocabulary(dataset);
     if (vocabulary.empty()) {
         std::cerr << "Vocabulary is empty. Cannot proceed. Check tokenization or data." << std::endl;
         return 1;
     }
    std::cout << "Vocabulary size: " << vocabulary.size() << std::endl;

    // --- TF-IDF Calculation ---
    std::cout << "Calculating Document Frequencies (DF)..." << std::endl;
    std::map<std::string, int> docFrequency;
    int N = dataset.size(); // Total number of documents
    for (const auto& dp : dataset) {
        std::vector<std::string> tokens = tokenize(dp.text);
        std::set<std::string> uniqueTokensInDoc(tokens.begin(), tokens.end());
        for (const std::string& token : uniqueTokensInDoc) {
            docFrequency[token]++;
        }
    }

    std::cout << "Calculating Inverse Document Frequencies (IDF)..." << std::endl;
    std::map<std::string, double> idfScores;
    for (const auto& pair : vocabulary) {
        const std::string& word = pair.first;
        int df = 0;
        auto df_it = docFrequency.find(word);
        if (df_it != docFrequency.end()) {
            df = df_it->second;
        }
        // Calculate IDF using log(N / (1 + df)) to avoid division by zero
        idfScores[word] = std::log(static_cast<double>(N) / (1.0 + static_cast<double>(df)));
    }
    // --- End TF-IDF Calculation ---

    std::cout << "Vectorizing data using TF-IDF..." << std::endl;
    std::unordered_map<std::string, int> label_to_id;
    int label_id_counter = 0;
    for (auto& dp : dataset) {
        dp.features = vectorize(dp.text, vocabulary, idfScores);
        if (label_to_id.find(dp.label) == label_to_id.end()) {
            label_to_id[dp.label] = label_id_counter++;
        }
    }

     if (label_id_counter == 0) {
          std::cerr << "No labels found in the dataset. Cannot train. Check JSON data." << std::endl;
          return 1;
     }


    std::cout << "Found " << label_id_counter << " unique labels." << std::endl;

    // Create one-hot encoded targets
     int num_labels = label_id_counter; // Use the counter which is the number of unique labels
    for (auto& dp : dataset) {
        dp.target.resize(num_labels, 0.0);
        int current_label_id = label_to_id[dp.label];
        if (current_label_id < num_labels) {
            dp.target[current_label_id] = 1.0;
        } else {
             std::cerr << "Error: Label ID mismatch during one-hot encoding for label: " << dp.label << std::endl;
             // Handle error appropriately, maybe skip this data point or exit
        }
    }

     // Create reverse mapping for inference output
     std::vector<std::string> id_to_label(num_labels);
     for(const auto& pair : label_to_id) {
         if (static_cast<size_t>(pair.second) < id_to_label.size()) {
            id_to_label[pair.second] = pair.first;
         } else {
             std::cerr << "Error: Label ID out of bounds when creating reverse map for label: " << pair.first << std::endl;
         }
     }


    // 3. Initialize Neural Network
    int input_size = vocabulary.size();
    int output_size = num_labels;
     if (input_size == 0 || output_size == 0) {
         std::cerr << "Error: Input or output size is zero. Cannot initialize network." << std::endl;
         return 1;
     }

    NeuralNetwork nn(input_size, HIDDEN_SIZE, output_size);
    std::cout << "Initialized Neural Network (Input: " << input_size
              << ", Hidden: " << HIDDEN_SIZE << ", Output: " << output_size << ")" << std::endl;

    // 4. Train Network
    std::cout << "Starting training (" << EPOCHS << " epochs)..." << std::endl;
    std::mt19937 rng(std::random_device{}()); // Random number generator for shuffling

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
         // Shuffle dataset for each epoch (important for SGD)
         std::shuffle(dataset.begin(), dataset.end(), rng);

         double total_epoch_loss = 0.0; // Track epoch loss

        for (const auto& dp : dataset) {
             // Basic check for valid feature vector size
             if (dp.features.size() != static_cast<size_t>(input_size)) {
                 std::cerr << "Warning: Skipping data point with unexpected feature size. Expected: "
                           << input_size << ", Got: " << dp.features.size() << std::endl;
                 continue; // Skip this data point
             }
             // Basic check for valid target vector size
              if (dp.target.size() != static_cast<size_t>(output_size)) {
                 std::cerr << "Warning: Skipping data point with unexpected target size. Expected: "
                           << output_size << ", Got: " << dp.target.size() << std::endl;
                 continue; // Skip this data point
             }

            nn.train(dp.features, dp.target);

             // Calculate loss for this data point
             std::vector<double> current_predictions = nn.predict(dp.features);
             total_epoch_loss += nn.calculate_mse(current_predictions, dp.target);
        }

        double average_epoch_loss = total_epoch_loss / dataset.size();
        if ((epoch + 1) % 10 == 0) { // Print progress every 10 epochs
            std::cout << "Epoch " << (epoch + 1) << "/" << EPOCHS << " completed. Avg Loss: " << average_epoch_loss << std::endl;
        }
    }
    std::cout << "Training finished." << std::endl;


    // 5. Inference Loop
    std::cout << "--- Text Classification Inference ---" << std::endl;
    std::cout << "Enter text to classify (or type 'quit' to exit):" << std::endl;
    std::string input_text;

    while (true) {
        std::cout << "> ";
        if (!std::getline(std::cin, input_text)) {
             if (std::cin.eof()) { // Handle Ctrl+D (end-of-file)
                 std::cout << "Exiting due to EOF." << std::endl;
                 break;
             }
             // Handle other potential stream errors
             std::cerr << "Error reading input. Exiting." << std::endl;
             break; // Exit on stream error
         }

        if (input_text == "quit") {
            break;
        }
        if (input_text.empty()) {
            continue; // Ignore empty input
        }

        // Preprocess user input using TF-IDF
        std::vector<double> input_features = vectorize(input_text, vocabulary, idfScores);

        // Predict
        std::vector<double> predictions = nn.predict(input_features);

        // Find the label with the highest probability
        int predicted_label_id = 0; // Default to 0
        // Find the index of the max element
        auto max_it = std::max_element(predictions.begin(), predictions.end());
        predicted_label_id = std::distance(predictions.begin(), max_it);
        double max_prob = *max_it;

        // Output result based on thresholds
        const double POSITIVE_THRESHOLD = 0.65; // Tightened threshold
        const double NEGATIVE_THRESHOLD = 0.35; // Tightened threshold

        std::string final_prediction = "uncertain";
        double confidence = max_prob; // Default confidence to the max probability

        // Assuming id_to_label[0] is positive and id_to_label[1] is negative
        if (output_size == 2 && static_cast<size_t>(predicted_label_id) < id_to_label.size()) {
            // Use the ID with the highest probability to check thresholds
            if (predicted_label_id == 0 && max_prob > POSITIVE_THRESHOLD) { // Predicted Positive is confident
                final_prediction = id_to_label[0];
            } else if (predicted_label_id == 1 && max_prob > POSITIVE_THRESHOLD) { // Predicted Negative is confident (using its probability)
                 final_prediction = id_to_label[1];
            } else if (predicted_label_id == 0 && max_prob < NEGATIVE_THRESHOLD) { // Predicted Positive has low confidence -> Negative
                 final_prediction = id_to_label[1];
            } else if (predicted_label_id == 1 && max_prob < NEGATIVE_THRESHOLD) { // Predicted Negative has low confidence -> Positive
                 final_prediction = id_to_label[0];
            } // Otherwise, it remains "uncertain"

        } else { // Fallback for non-binary or error cases
             if (predicted_label_id != -1 && static_cast<size_t>(predicted_label_id) < id_to_label.size()) {
                 final_prediction = id_to_label[predicted_label_id]; // Use raw prediction
             } else {
                 std::cerr << "Error: Could not determine predicted label or label ID out of bounds." << std::endl;
                 final_prediction = "error";
             }
        }


        std::cout << "Predicted: " << final_prediction
                  << " (Confidence: " << confidence << ")" << std::endl;

    }

    std::cout << "Exiting program." << std::endl;
    return 0;
} 