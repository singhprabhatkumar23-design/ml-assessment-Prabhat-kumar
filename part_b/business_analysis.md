
# B1. Problem Formulation
(a) ML Problem Formulation

Target Variable: items_sold (Continuous numerical value).

Candidate Input Features: * Store attributes: store_id, location_type (urban/rural), store_size, competition_density.

Promotion details: promotion_type (BOGO, Flat Discount, etc.).

Temporal features: month, is_weekend, is_festival, footfall.

ML Problem Type: Regression.

Justification: The goal is to predict a specific quantity (volume of sales) to determine which promotion maximizes that number. Since the output is a continuous value, regression is the appropriate approach.

(b) Target Variable Selection: Items Sold vs. Revenue
Using items_sold (volume) is more reliable because total revenue is often conflated by the promotion type itself (e.g., a "Flat 50% Discount" might double the items sold but result in the same revenue as a "No Promotion" day).

Principle: This illustrates Target Independence. In ML, you should choose a target that directly reflects the consumer behavior you want to influence (demand) rather than a metric heavily influenced by the price fluctuations of the features (promotions) themselves.

(c) Modeling Strategy: Global vs. Local
Instead of one global model, I propose a Clustered Modeling Strategy or Hierarchical Modeling.

Justification: Since urban and rural stores behave differently, we can train separate models for different location_type segments. This allows the model to learn specific coefficients for rural price sensitivity versus urban brand loyalty, which a single global model might "average out" and ignore.

# B2. Data and EDA Strategy
(a) Data Joining and Grain

Joining: Join transactions with store_attributes on store_id, promotion_details on promotion_id/type, and calendar on transaction_date.

Grain: The final grain should be Store-Month-Promotion Type.

Aggregations: Sum the items_sold and average the footfall and competition_density for each store per month.

(b) EDA Strategy

Boxplot of items_sold by promotion_type: To see which promotion has the highest median impact and identify outliers.

Time-Series Line Chart (Sales vs. Month): To identify seasonality (e.g., spikes during festivals) which influences feature engineering for holiday flags.

Heatmap of location_type vs. promotion_type: To check for interactions—does BOGO work better in urban areas than rural ones?

Scatter Plot of competition_density vs. items_sold: To see if highly competitive areas require more aggressive promotions to move inventory.

(c) Handling Data Imbalance (80% No Promotion)
An 80% "No Promotion" rate means the model might become biased toward predicting baseline sales.

Steps to address: I would use Downsampling of the "No Promotion" days to balance the dataset or, more effectively, create a Binary Flag (Is_Promoted) and use Weighted Loss Functions to penalize errors on promotion days more heavily, ensuring the model learns the "lift" created by discounts.

# B3. Model Evaluation and Deployment
(a) Train-Test Split and Metrics

Setup: Use a Time-Based Split (e.g., train on the first 30 months, test on the last 6 months).

Why avoid Random Split? Random splitting leads to "data leakage" where future information helps predict the past, rendering the model useless for real-world forecasting.

Metrics: * RMSE (Root Mean Squared Error): To penalize large prediction errors.

MAE (Mean Absolute Error): To understand the average "off" count in items sold per store.

(b) Investigating Model Recommendations
Using SHAP (SHapley Additive exPlanations) or Feature Importance, I would show the marketing team that in December, the is_festival and is_month_end features have high importance, making "Loyalty Points" effective for gift shoppers. In March, perhaps competition_density or "End of Season" trends make "Flat Discounts" more necessary to clear stock. This provides "Why" instead of just "What."

(c) Deployment and Monitoring

Saving: Save the model using joblib or pickle and wrap it in a Flask/FastAPI service.

Preparation: New data (upcoming month’s calendar and store traits) is passed through the same ColumnTransformer pipeline.

Monitoring: Implement Data Drift and Model Drift detection. If the RMSE on actual sales starts to climb (e.g., due to a change in consumer trends or inflation), it triggers an automated retraining pipeline.