# Random Forest From Scratch

This project implements a Random Forest algorithm from scratch in Python, without using high-level machine learning libraries. It demonstrates the core concepts behind ensemble learning and decision trees.

## Project Structure

- `main.py` — Main script to run the Random Forest implementation.
- `Data/` — Contains the dataset files:
  - `train.csv` — Training data (Titanic dataset).
  - `test.csv` — Test data.
  - `gender_submission.csv` — Sample submission file.

## Features
- Custom implementation of Random Forest algorithm
- Handles CSV data loading and preprocessing
- Predicts survival on the Titanic dataset

## Getting Started

### Prerequisites
- Python 3.7+
- (Recommended) Create and activate a virtual environment:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- Install any required packages (if used):
  ```bash
  pip install -r requirements.txt
  ```

### Running the Project

1. Place the dataset files in the `Data/` directory.
2. Run the main script:
   ```bash
   python main.py
   ```

## Dataset
This project uses the Titanic dataset from Kaggle. Make sure the following files are in the `Data/` folder:
- `train.csv`
- `test.csv`
- `gender_submission.csv`

## Project Goals
- Learn and demonstrate how Random Forest works internally
- Practice implementing machine learning algorithms from scratch

## References
- [Random Forest (Wikipedia)](https://en.wikipedia.org/wiki/Random_forest)
- [Titanic: Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/c/titanic)

---
Feel free to modify and extend this project for your own experiments!
