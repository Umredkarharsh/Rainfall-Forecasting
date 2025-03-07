
# Mumbai Rainfall Forecasting

## Introduction

Accurate rainfall forecasts are essential for Mumbai's water resource management. This case study examines the development of a machine learning model to predict monthly rainfall in Mumbai, leveraging historical data to improve water allocation, storage, and distribution strategies. Effective rainfall prediction can lead to optimized reservoir management, reduced operational costs, and a more consistent water supply throughout the year.

#### Problem Statement
The challenge is to develop a predictive model that can accurately forecast monthly rainfall in Mumbai. 
This model will assist water authorities in making informed decisions regarding:
•	Reservoir levels and releases.
•	Water distribution planning.
•	Drought mitigation strategies.
•	Flood control measures.

#### Data examines
This dataset total is based on a location in Mumbai and we see here these datasets are from 1901 to 2021 total is 121 years from 1901, with monthly and yearly precipitation patterns from 1901 to 2021 utilized for forecasting upcoming rain sessions of 2025 we lack some years  

 Precipitation data was analyzed by categorizing it into four seasons:    
(1) Non-monsoon (January, February, March)   
(2) Pre-monsoon (April, May)   
(3) Southwest monsoon (June, July, August, September)  
(4) Northeast monsoon (October, November, and December)    
then basic statistical analysis finds out the maximum and minimum rainfall, mean, standard deviation, skewness, and Kurtosis.  

#### Asking Questions(Ask)
* What questions should we ask BMC and stakeholders to learn more about the dataset, its limitations, requirements, and results?   
* Why is there so much rain in Mumbai?  
- Why is Mumbai suffering from a water crisis?  
- How much water is needed in Mumbai in a year?  
- How much water is supplied in Mumbai in the year?  
- What is the potential for rainwater harvesting to supplement  Mumbai's water supply, given the historical rainfall patterns?  
- What other data would be needed to complement the rainfall data and make it more useful for planning purposes (e.g., land use data, population data, evaporation rates)?  
- How could the historical rainfall dataset be used to inform urban planning decisions related to water management?  
- Is the BMC open to sharing its own rainfall data or other relevant water resources data with researchers or the public?  
- What are the known limitations in the accuracy of current rainfall forecasts?

## Data Preprocessing(Prepare,Process,Analysis,Share)
Data Cleaning & Preprocessing: Handling missing values, outliers, and feature engineering.  
🔹 **Data Preprocessing:** Dectect outliers and treat them with  the help of IQR technique, Capping technique, checking Null value, duplicate values, Data relationship.

🔹 **Exploratory Data Analysis (EDA):** Visualizing data patterns and insights using matplotlib, Seaborn, Plotly libarires.categorizing it into four season(Winter,Summar,Post-Monsoon,Monsoon) yearly rainfall classification.

🔹 **Feature Engineering:** Converts categorical text values into numerical labels for better analysis, visualization, and machine-learning compatibility use map() function. we use the Standard Scaling technique (Z-score normalization) to numerical features in the dataset.  

🔹 **Model Building & Evaluation:** Implementing Machine Learning Regression Models(Linear Regression,Lasso,Ridge,Random Forest Regressor,Decision Tree Regressor,Gredaint Boosting Regressor,AdaBoost Regressor,XGB Regressor,). split Dataset into Train Test with 20% rate. Hyperparameter tuning  are using with models are (XGBoost Regressor,Gredaint Boosting, Decision Tree Regressor,Random Forest Regressor). Based on the observed issues with the XGBoost hyperparameter model trained and the strong performance of Ridge Regression, Ridge Regression is the best model for this dataset. Based on the observed issues with the XGBoost hyperparameter model trained and the strong performance of Ridge Regression, Ridge Regression is the best model for this dataset.with overffiting problem.


🔹 **Snap Technique:** The SHAP plot now highlights the strong influence of the monsoon season and the dry season on the model's output. The insights are consistent with Mumbai's tropical monsoon climate and the significant impact of rainfall and humidity on the predicted variable. 


🔹 **Deployment:** Deploying the model using Streamlit  for real-world applications.But Facing issues and error of requirements.txt file.

## **Results & Insights**  

Based on the insights from both analyses:

Based on the observed issues with the XGBoost hyperparameter model trained and the strong performance of Ridge Regression, Ridge Regression is the best model for this dataset.  
Unfortunatly Rigde Regression model was Overffited Streamlit deployment have succssfully deploy check it out - https://mumbai-raifall.streamlit.app/ 
next goal toward hyperparameter and feature engineering to resolve overfitting problem

## **Acknowledgments**  

BMC Weather Reports, charts, and case papers from my internship company helped me answer all my questions, providing more clarity throughout my project work. 
