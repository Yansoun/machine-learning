🎯 A/B Test Evaluation – Ad vs PSA Conversion Impact

An end-to-end data analysis project that measures the effectiveness of advertising compared to public service announcements (PSA) using statistical testing.

🚀 Overview

The goal of this project is to determine whether showing ads leads to a higher conversion rate than showing PSA content.
This helps marketing and product teams make data-driven decisions about where to invest resources.

📊 Dataset

Source: Kaggle – A/B Advertising Dataset
Description:

~580,000 user interactions

Columns include:

group (ad or psa)

converted (True/False)

total ads shown

most ads day

most ads hour

user id

Key preprocessing steps:

Removed unused index column

Standardized column names

Calculated conversion rates by group

Validated sample distribution (ad vs psa)

Prepared data for z-test and confidence intervals

🧩 Methodology & Statistical Testing

To evaluate whether the difference between ad and psa groups is statistically significant, the following statistical steps were performed:

Methods used:

Conversion Rate Calculation

Difference in Means

95% Confidence Interval (CI)

Two-Proportion Z-test

P-value Analysis

Results:

Ads had a higher conversion rate than PSA

Z-test statistic was significantly positive

p-value < 0.05, meaning the difference is statistically significant

The 95% confidence interval did not include zero, confirming a real effect

Conclusion:

➡️ Ads significantly increase conversion compared to PSA.
➡️ Ads should be preferred if the business goal is maximizing conversions.

💻 Tech Stack

Python

Libraries: Pandas, NumPy, Statsmodels, Matplotlib, Seaborn, SciPy

⚙️ How to Run Locally
git clone https://github.com/Yansoun/machine-learning/tree/project7_ab_test.git
cd ab-test-evaluation
pip install -r requirements.txt
jupyter notebook
