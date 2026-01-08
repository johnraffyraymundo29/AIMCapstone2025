# AIM ML Capstone — MovieLens 1M

## Project Overview
This project demonstrates an end-to-end machine learning lifecycle for building a
fair and explainable recommendation system using the MovieLens 1M dataset.

It covers:
- Problem framing and business context
- Data preprocessing and exploratory data analysis
- Recommendation model development and evaluation
- Explainability and fairness auditing
- Responsible use of Generative AI

## Dataset
- **Source:** MovieLens 1M (GroupLens Research)
- **Domain:** Recommender Systems / eCommerce-style personalization
- **Data includes:** user–item interactions, movie metadata, and user demographics

> The dataset is publicly available and licensed for academic use.

## Repository Structure
## Notebooks Overview

| Notebook | Description |
|--------|------------|
| 01_problem_framing_and_data_understanding.ipynb | Business problem definition and overview of the MovieLens 1M dataset |
| 02_eda_and_feature_analysis.ipynb | Exploratory data analysis and dataset insights |
| 03_MF_Train_eval.ipynb | Baseline RMSE and Matrix Factorization (TruncatedSVD) model training and evaluation |
| 04_recommendation_generation_and_qualitative_evaluation.ipynb | Top-K recommendation generation and qualitative evaluation |
| 05_fairness_and_explainability.ipynb | Fairness analysis, explainability discussion, and ethical considerations |

## Model Evaluation Summary

Model performance is evaluated using Root Mean Squared Error (RMSE).

- A global mean baseline model is used as a performance benchmark.
- A Matrix Factorization model (TruncatedSVD) demonstrates improved RMSE on both validation and test sets.
- Qualitative evaluation of Top-10 recommendations confirms that predicted rankings are coherent, interpretable, and plausible for real users.

This qualitative check complements RMSE-based evaluation by confirming that the ranked outputs are meaningful beyond numerical metrics.

## Model Artifacts

Large model artifacts (`pred_df.parquet`, `movie_map.parquet`) are not
committed to GitHub due to size limitations.

To generate them:
1. Run Notebook 03 (Model Training & Evaluation)
2. Run Notebook 04 (Recommendation Generation)

These notebooks will output the required parquet files for local deployment.


## How to Run

All notebooks are designed to run in Google Colab.

1. Upload the MovieLens 1M dataset files (`ratings.dat`, `movies.dat`, `users.dat`) to `/content/`
2. Run notebooks sequentially from Notebook 01 to Notebook 05
