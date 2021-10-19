# Udacity_Data_Scientist_Starbucks_Capstone_Project
Make recommendation system for Starbucks Offer

## Project Overview
This project is Udacity Starbucks Capstone Challenge in Data Scientist Course. I made a machine learning model which predicted offer completed for each customer based on user profiels, offer portfolio and transcript data. Based on the ML model, I made a recommendation web application for Starbucks on how to recommend offer to a user.

**Data sets, Data cleaning, exploration, visualization and modeling** please refer to [Starbucks_Capstone_notebook.ipynb](Starbucks_Capstone_notebook.ipynb)

## Summary
Random Forest model has the best performance among those 6 models (Decision Tree, Support Vector Machine, Naive Bayes, Random Forest, K-Nearest Neighbors, LogisticRegression), I use it to recommend offer for users.

# How to Run the Recommondation web application

## Install Apps
1. Anaconda
2. Python 3

## Steps to setup environment
1. Open Terminal
2. Go to project directory
```
cd project_path
```
3. Create virtual python env
```
python3 -m venv env_name
```
4. Activate the virtual env
```
source env_name/bin/activate
```
5. Install necessary packages
```
pip install flask pandas plotly gunicorn nltk sklearn sqlalchemy
```
6. Go to workspace folder
```
cd workspace
```
Now the environment is ready to run the app.

# Step to run the web application
1. Data processing
```
python data/process_data.py data/portfolio.json data/profile.json data/transcript.json data/user_offer_matrix.db
```
2. Model training
```
python models/train_classifier.py data/user_offer_matrix.db models/recommendation.pkl
```
3. Run the app
```
python app/run.py
```
4. Go to http://localhost:3001/ to check the result
<img src="https://user-images.githubusercontent.com/8360742/137706110-4b66c4bc-fb46-4cea-9b79-6b8d5143684d.png" width="800" />
6. User info input example: {'gender':1, 'age':35,'income':50000, 'memberdays':1}
7. Output is the probability of user completed each offer:
<img src="https://user-images.githubusercontent.com/8360742/137703150-9fe8191f-de0c-4314-b29b-13250d4aee32.png" width="500" />

