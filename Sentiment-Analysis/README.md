# Sentiment Analysis with a Transformer

This project documents the process of building, training, and—most importantly—properly evaluating a Transformer-based neural network for sentiment analysis. The model is built from scratch using PyTorch and classifies sentences as "Positive" or "Negative".

This repository doesn't just contain the final code; it represents a complete learning journey through the common pitfalls and advanced concepts of a real-world machine learning workflow.

## 🤖 Project Overview & Model Architecture

The core of this project is the `SentimentTransformer` model, which was built using the following PyTorch components:

1. **Word Embeddings (`nn.Embedding`):** The first layer, which converts numeric word indices from our vocabulary into dense vector representations.
2. **Positional Encoding (`PositionalEncoding`):** A custom class that injects information about word order into the embeddings. This is critical because Transformers, by default, don't know the sequence of words.
3. **Transformer (`nn.TransformerEncoder`):** The "brain" of the model. We use the built-in PyTorch `TransformerEncoderLayer` and `TransformerEncoder` modules, which contain the multi-head attention mechanisms and feed-forward networks.
4. **[CLS] Token:** A special token, `[CLS]`, was added to the beginning of every sentence. The Transformer is designed to "read" the whole sentence and store a summary of its meaning in this token's final output vector.
5. **Classification Head (`nn.Linear`):** A single fully-connected layer that takes the final output vector of the `[CLS]` token and maps it to our two output classes ("Positive" and "Negative").

---

## 💡 My Learning Journey & Challenges Faced

Building the model was only the first step. The most valuable part of this project was learning how to **properly validate** it and avoid common traps.

### Challenge 1: The 100% Accuracy "Trap" (Overfitting)

- **Problem:** My first successful training run! The model trained for 200 epochs and achieved 100% accuracy. This *felt* like a perfect success, but it was a major red flag. The model was being evaluated on the **exact same data** it was trained on. It had simply memorized the answers.
- **Solution: The Train/Test Split:** I learned to use `sklearn.model_selection.train_test_split` to split the full dataset into two parts:
    - **`train_data` (e.g., 80%):** The *only* data the model is allowed to learn from.
    - **`test_data` (e.g., 20%):** Kept separate as an "unseen final exam" to get a realistic score. This immediately revealed the model's true performance (which was not 100%).

### Challenge 2: The "Unlucky" Split (Class Imbalance)

- **Problem:** I realized a new potential issue: what if the *random* split accidentally put 70% of the "Positive" examples in the training set and only 30% in the test set? The model would be trained on a biased dataset and tested on a differently biased one, leading to a misleading score.
- **Solution: Stratified Splitting:** The solution was to use the `stratify` parameter. By passing my list of `labels` to `train_test_split(..., stratify=labels)`, I ensured that both the `train_data` and `test_data` had the **exact same percentage** of "Positive" and "Negative" examples as the original dataset.

### Challenge 3: Vocabulary and "Data Leakage"

- **Problem:** A logical question came up: "Should I build my vocabulary from *all* the data (train and test)?" It seems to make sense, as the model should "know" all the words it might see.
- **Solution: Preventing Data Leakage:** This is a classic and subtle bug called **data leakage**. The vocabulary must be built **only** from the `train_data`.
    - **Why?** The `test_data` simulates the real world, where your model will encounter words it has *never seen before*. By building the vocab only from `train_data`, any new words in the `test_data` get mapped to our `<PAD>` token (index 0). This forces the model to learn to be robust and make predictions even with unknown words.
    - Giving the model the test set's words to build a vocab is like giving a student the vocabulary list from the final exam before they've even started the class.

### Challenge 4: Moving Beyond "Accuracy"

- **Problem:** "Accuracy" seems like the only metric that matters, but it can be deeply misleading. If a test set has 95 "Negative" reviews and 5 "Positive" reviews, a lazy model that *always* guesses "Negative" will have **95% accuracy** but be completely useless.
- **Solution: A Full Evaluation Report:** I learned to use `sklearn.metrics` to generate a **Confusion Matrix** and a **Classification Report**. These metrics give a complete picture of performance:
    - **Confusion Matrix:** A 2x2 grid that shows *exactly* where the model is getting confused (e.g., how many *actual* positives it *falsely* predicted as negative).
    - **Precision:** Of all the times the model *predicted* "Positive," what percentage was *actually* correct? (Measures "trustworthiness").
    - **Recall:** Of all the *actual* "Positive" reviews, what percentage did the model successfully *find*? (Measures "thoroughness").
    - **F1-Score:** The harmonic mean of Precision and Recall. This is often the single best metric for a model's performance, as it balances both.

---

## 🏁 Final Status

This project is now a complete pipeline that correctly prepares data, builds a Transformer model, and—most importantly—uses robust, professional techniques to validate its performance and avoid common data science pitfalls.
