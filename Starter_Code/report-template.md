# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.

The purpose of the analysis is to build and compare machine learning models to predict loan risk using financial information.

* Explain what financial information the data was on, and what you needed to predict.

The data contains information on loans, including loan amount, interest rate, borrower income, number of accounts, and other financial information.

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).

The task is to predict the risk of a loan, where a loan is classified as high risk if it is likely to default, and low risk otherwise.

* Describe the stages of the machine learning process you went through as part of this analysis.

The stages of the machine learning process include data cleaning and preprocessing, splitting data on features and labels, creating the test and trainig data sets, model selection, model evaluation, and performance comparison.

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

There were 2 methods that were used and they are logistic regression with original data and logistic regression with resampled data using RandomOverSampler

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.

Accuracy: Our model has a high accuracy of 0.94, which means it correctly classifies approximately 94% of all the loans in our dataset. However, high accuracy can be misleading in imbalanced datasets, so it's essential to consider other metrics.

Precision: The precision of our model is approximately 0.87. This means that when our model predicts a loan as "at risk," it's correct about 87% of the time. In other words, the model has a relatively low false positive rate, making it reasonably reliable when it predicts a loan as risky.

Recall: The recall of our model is approximately 0.89. This indicates that our model is capable of capturing 89% of the actual risky loans in the dataset. It has a relatively low false negative rate, meaning it doesn't miss many risky loans.


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  
Accuracy: Our secondary model has an extremely high accuracy of 0.994, which means it correctly classifies approximately 99.4% of all the loans in our dataset. This is a very high accuracy rate, indicating that the model's overall performance is excellent.

Precision: The precision of our secondary model is approximately 0.994. This means that when our model predicts a loan as "at risk," it's correct about 99.4% of the time. The model has a very low false positive rate, which suggests it is highly reliable when it predicts a loan as risky.

Recall: The recall of our secondary model is approximately 0.994. This indicates that our model is capturing 99.4% of the actual risky loans in the dataset. It has an extremely low false negative rate, meaning it rarely misses risky loans.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

Based on the findings, the logistic regression model developed with resampled data (Model 2) outperforms the model created using the original data (Model 1), particularly in its ability to predict high-risk loans. Model 2 showcases notably elevated precision and recall scores for high-risk loans, a pivotal factor in minimizing potential financial losses for the lending institution. It is strongly advised to leverage the logistic regression model trained with resampled data (Model 2) for the purpose of credit risk analysis. This choice is justified by its substantial enhancement in accurately predicting high-risk loans when compared to the original model. By employing this model, the company can effectively evaluate loan applications and make well-informed choices during the loan approval process, thereby effectively mitigating credit risk.
