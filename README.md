# Twitter Sentiment and Engagement Analysis

## Description

This project provides a comprehensive analysis of Twitter data, focusing on sentiment classification. It leverages natural language processing (NLP) techniques and a sophisticated deep learning model to classify tweet polarity into **Positive**, **Negative**, or **Neutral** categories, achieving a final **test accuracy of approximately 83%**.

This repository is structured into three key areas:
1.  **Practical Application (`twitter_analytics.ipynb`):** A deep learning model (CNN + Bi-LSTM) is built and trained to perform sentiment analysis on a large dataset of tweets. This is the core of the project.
2.  **Foundational Concepts (`MachineLearning.ipynb`):** This notebook explores the core algorithms that underpin the main analysis. It serves as a conceptual guide to the "how" and "why" behind the more complex model.
3.  **Initial Exploration (`SentimentAnalysis.ipynb`):** This notebook contains the initial data cleaning and basic sentiment analysis work that served as a precursor to the deep learning model.

The concepts in `MachineLearning.ipynb` directly relate to the steps in `twitter_analytics.ipynb`:

*   **Optimization (Genetic & Simplex Algorithms vs. Adam Optimizer):** The genetic and simplex algorithms in `MachineLearning.ipynb` demonstrate how a program can search for an optimal solution. This is analogous to the `model.fit()` step in `twitter_analytics.ipynb`, where the `adam` optimizer works to find the best possible weights for the neural network to minimize classification error.
*   **Data Representation (Matrices & Vectors):** `MachineLearning.ipynb` explains how vectors and matrices work. In `twitter_analytics.ipynb`, this concept is applied when tweets are converted into numerical vectors and then into an `embedding_matrix` using Word2Vec. This matrix is the numerical representation of the vocabulary that the neural network can process.
*   **Problem Solving (Dynamic Programming):** The Knapsack problem in `MachineLearning.ipynb` is solved using dynamic programming, which breaks a large problem into smaller pieces. This mirrors the training process of a neural network, which learns iteratively, layer by layer and batch by batch, to solve the complex problem of classifying sentiment.

## Key Features & Model Architecture

*   **High-Accuracy Deep Learning Model:** The `twitter_analytics.ipynb` notebook implements a sophisticated model using a Convolutional Neural Network (CNN) combined with two Bidirectional LSTM layers.
*   **Custom Word Embeddings:** The project uses `gensim`'s Word2Vec to create custom word embeddings from the training data, allowing the model to understand the semantic context of words specific to the Twitter dataset.
*   **Optimized Training:** The model uses modern callbacks like `EarlyStopping` to prevent overfitting and `ReduceLROnPlateau` to dynamically adjust the learning rate, ensuring an efficient and effective training process.
*   **Machine Learning Exploration:** The `MachineLearning.ipynb` notebook serves as a practical guide to several optimization algorithms, showcasing their implementation with Python libraries like NumPy and SciPy.

## Getting Started

### Prerequisites

To run these notebooks, you will need a Python environment with the required libraries. The project was built using a Conda environment. All necessary packages are listed in the first cell of the `twitter_analytics.ipynb` notebook.

### Installation & Execution

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/csalaam/twitter_analytics.git
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd twitter_analytics
    ```
3.  **Set up Data**: Create a `data` folder in the root of the project directory. Place your `twitter_trainingdata.csv` and `twitter_testdata.csv` files inside this `data` folder.
4.  **Launch Jupyter Notebook:**
    ```sh
    jupyter notebook
    ```
5.  Open and run the cells in `twitter_analytics.ipynb`.

## Future Improvements

This project provides a solid foundation, and here are some ways it could be improved:

*   **Add Unit Tests:** The project currently lacks automated tests. Adding a test suite for the data cleaning and preprocessing functions would improve reliability.
*   **Refactor into Scripts:** The logic within the Jupyter notebooks could be refactored into reusable Python scripts. This would make it easier to run the analysis or train the model from the command line and prepare it for deployment.
*   **Error Handling:** Adding checks for file existence and more descriptive error messages would make the project more robust.
*   **Hyperparameter Tuning:** The deep learning model could be further optimized by systematically tuning hyperparameters like learning rate, dropout, and the number of neurons in each layer using tools like KerasTuner or Optuna.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.