# LLM_spam_detection

This project benchmarks various instruction-tuned Large Language Models (LLMs) on a zero-shot spam detection task. We evaluate each model's performance using two key metrics: Accuracy and Average Log Loss.

Evaluation Metrics

* Accuracy: This is the most straightforward metric. It measures the percentage of spam messages the model correctly classifies as spam. While easy to understand, it's a binary measure that treats a confident correct answer the same as a hesitant one.
* Average Log Loss (Cross-Entropy): This is a more powerful metric that evaluates the model's confidence. It heavily penalizes models that are "confidently wrong" and rewards models that are confident in the correct answer. A lower log loss score is better, with a score near 0.0 indicating that the model was consistently and highly confident in its correct predictions.

Together, these metrics provide a comprehensive view: Accuracy tells us the rate of correct classifications, while Log Loss reveals the model's underlying certainty and reliability.

Results:

==================== FINAL RESULTS SUMMARY ====================

Model: google/gemma-3-4b-it
  - Average Log Loss: 0.0000
  - Accuracy: 100.00% (12/12)
------------------------------
Model: google/gemma-3-270m-it
  - Average Log Loss: 0.0534
  - Accuracy: 100.00% (12/12)
------------------------------
Model: google/gemma-3-1b-it
  - Average Log Loss: 0.3229
  - Accuracy: 83.33% (10/12)
------------------------------
