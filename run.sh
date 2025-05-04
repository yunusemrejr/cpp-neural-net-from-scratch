#!/bin/bash

# Script to compile and run the C++ text classifier
set -euo pipefail

 
echo -e "\033[2J\033[H" # Clear screen
echo "🔸🔸🔸 Yun ^...^ f(x) Presents 🔸🔸🔸"
sleep 0.4
echo "C++ Text Classifier Neural Network"
echo "        _____"
sleep 0.2
echo "      .-'.  ':'-."
sleep 0.2
echo "    .''::: .:    '."
sleep 0.2
echo "   /   :::::'      \\"
sleep 0.2
echo "  ;.    ':' \`       ;"
sleep 0.2
echo "  |       '..       |"
sleep 0.2
echo "  ; '      ::::.    ;"
sleep 0.2
echo "   \\       '::::   /"
sleep 0.2
echo "    '.      :::  .'" 
sleep 0.2
echo "      '-.___'_.-'"
sleep 1
echo -e "\033[2J\033[H" # Clear screen again



CPP_FILE="main.cpp"
JSON_DATA_FILE="data.json"
EXECUTABLE="classifier"

# --- Check Dependencies ---

# Check for C++ compiler (g++ or clang++)
COMPILER=""
if command -v clang++ &> /dev/null; then
    COMPILER="clang++"
elif command -v g++ &> /dev/null; then
    COMPILER="g++"
else
    echo "Error: No C++ compiler (clang++ or g++) found. Please install one." >&2
    exit 1
fi
echo "Using compiler: $COMPILER"

# Check if source file exists
if [ ! -f "$CPP_FILE" ]; then
    echo "Error: Source file '$CPP_FILE' not found." >&2
    exit 1
fi

# Check if data file exists
if [ ! -f "$JSON_DATA_FILE" ]; then
    echo "Error: Data file '$JSON_DATA_FILE' not found." >&2
    exit 1
fi

# --- Compilation ---
echo "Compiling $CPP_FILE..."

# Compile with C++17 standard, optimizations, and warnings
"$COMPILER" -std=c++17 -O2 -Wall -Wextra -pedantic "$CPP_FILE" -o "$EXECUTABLE"

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed." >&2
    exit 1
fi

echo "Compilation successful. Executable created: $EXECUTABLE"

# --- Execution ---
echo "Running the classifier..."

# Execute the compiled program, passing the data file as an argument
./"$EXECUTABLE" "$JSON_DATA_FILE"

# Check execution status (optional, as the program might run indefinitely until user quits)
if [ $? -ne 0 ]; then
    echo "Warning: Program exited with a non-zero status." >&2
    # Don't exit the script here, as non-zero might be intentional or from user action
fi

echo "Classifier finished or was terminated."

exit 0 