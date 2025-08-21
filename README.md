# LLM_spam_detection

This project benchmarks various instruction-tuned Large Language Models (LLMs) on a zero-shot spam detection task. We evaluate each model's performance using two key metrics: Accuracy and Average Log Loss.

Evaluation Metrics

* Accuracy: This is the most straightforward metric. It measures the percentage of spam messages the model correctly classifies as spam. While easy to understand, it's a binary measure that treats a confident correct answer the same as a hesitant one.
* Average Log Loss (Cross-Entropy): This is a more powerful metric that evaluates the model's confidence. It heavily penalizes models that are "confidently wrong" and rewards models that are confident in the correct answer. A lower log loss score is better, with a score near 0.0 indicating that the model was consistently and highly confident in its correct predictions.

Together, these metrics provide a comprehensive view: Accuracy tells us the rate of correct classifications, while Log Loss reveals the model's underlying certainty and reliability.

Results:

Of course. Here are your results formatted into a clean Markdown table, perfect for a `README.md` file.

I've highlighted the best score in each column to make the top performers stand out.

***

### LLM Spam Detection Benchmark Results

The table below summarizes the performance of various Gemma models on a zero-shot spam detection task. The models are ranked by their **F1-Score**.

* For **F1-Score, Accuracy, Precision, and Recall**: Higher is better ⬆️
* For **Avg Log Loss**: Lower is better ⬇️

| Model | F1-Score | Accuracy | Precision | Recall | Avg Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `google/gemma-3n-e4b-it` | **92.31%** | **92.00%** | **85.71%** | 100.00% | 0.8017 |
| `google/gemma-3-4b-it` | 88.89% | 88.00% | 80.00% | 100.00% | 1.2733 |
| `google/gemma-2-2b-it` | 88.89% | 88.00% | 80.00% | 100.00% | **0.7990** |
| `google/gemma-3-1b-it` | 64.86% | 48.00% | 48.00% | 100.00% | 5.7998 |
| `google/gemma-3-270m-it`| 64.86% | 48.00% | 48.00% | **100.00%** | 1.6349 |
