# Relationship Manager Portfolio Recommender Model

## Dash app for Community


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dash-v2-orange)](https://dash.plotly.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24-orange.svg)](https://scikit-learn.org/stable/)
[![Made with ML](https://img.shields.io/badge/Made%20with-ML-red)](https://github.com/madewithml)
[![Recommendation Engine](https://img.shields.io/badge/Type-Recommendation%20Engine-brightgreen)](https://en.wikipedia.org/wiki/Recommender_system)
[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)                                                                                                                 

This Dash web application provides personalized portfolio recommendations for relationship managers (RMs) based on customer attributes and behavior. It uses a combination of LightGBM and Nearest Neighbors (NN) algorithms for recommendations.

[![Dental Diagnosis Demo Video](https://github.com/dallo7/mcbT/blob/27605c9f85a9112cbc3a193fc077dfc770c1abe6/reco.png)


The app is hosted on render.com https://mcbrecommender.onrender.com/  (when you click the link give it at least a minute for the service to restart)                    
  

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Models & Data](#models--data)
- [Contributing](#contributing)

## Project Overview

**Purpose:** The RM Portfolio Recommender assists relationship managers in suggesting the most suitable financial products (accounts, loans, investment plans) to their customers, thereby potentially increasing customer engagement and satisfaction.

**Target Users:**

- **Relationship Managers:**  Can use this tool to quickly generate data-driven recommendations tailored to individual customers.
- **Financial Institutions:** Can integrate this tool into their CRM systems to enhance the advisory services provided by RMs.

## Features

- **Personalized Recommendations:**  Utilizes customer attributes (age, income, transaction behavior, savings) and historical data to generate personalized recommendations.
- **Multiple Recommendation Models:**
    - **LightGBM:** A gradient boosting framework known for its speed and accuracy.
    - **Nearest Neighbors:**  Finds similar customers and recommends products based on their choices.
- **Interactive Dashboard:** A user-friendly interface to input customer data and view recommendations.
- **Explanation Visualization:**  Bar graph displaying the similarity of recommended accounts to the customer's profile.
- **Recommendation Tracking:** Logs customer data and recommendations for analysis and auditing.

## Technologies

- **Dash:** Python framework for building web applications.
- **Dash Bootstrap Components (dbc):** For styling and layout.
- **Pandas:** Data manipulation and analysis library.
- **scikit-learn:** For machine learning (Nearest Neighbors, StandardScaler).
- **LightGBM:** Gradient boosting framework for machine learning.
- **Plotly:** For data visualization (bar graph).
- **Joblib:** For saving and loading the LightGBM model.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

## Install Dependencies:

    ```Bash
    pip install -r requirements.txt
    Use code with caution.
    ````                                                                                 
  
~Make sure you have the necessary model files (mrLgbm.pkl) and the dataset (mcb2.csv) in the correct locations.
Run the App:~                                        
  
```Bash The app should be accessible at http://127.0.0.1:5050/ ```

2. Usage
 * Access the App: Open your web browser and navigate to the app's URL.
 * Input Customer Data: Enter the required information about the customer (age, salary, transactions, savings, customer market).
 * Click "Recommend Accounts": The app will display the top 3 recommended accounts, along with an explanation bar graph and a table logging the recommendations.
   
3. File Structure
 * app.py: Main Dash application code.
 * mcb2.csv: Dataset for training and recommendation (replace with your actual data).
 * mrLgbm.pkl: Saved LightGBM model file (replace with your actual model).
   
4. Models & Data
 * LightGBM Model: Trained on customer data to predict suitable accounts.
 * Nearest Neighbors: Finds similar customers based on their financial profiles.
 * Dataset: Customer data (mcb2.csv) containing attributes used for recommendations.
   
5. Contributing
Contributions are welcome! Feel free to open issues or pull requests to suggest improvements or fix bugs.

