# Multivariate-Regression-Analysis


## Overview
This theoretical project aims to optimize the performance of the Liftbot robot by predicting its project times using multivariate regression. The analysis focuses on how various parameters—such as the weight lifted, scaffold height, and scaffold length—affect the total time required for the robot to complete its tasks. The Liftbot, conceived by the company Kewazo, is designed to efficiently lift and lower weights along a rail attached to scaffolds, making it an essential tool in construction and maintenance. By leveraging AI-generated data, this project seeks to provide insights that can enhance the operational efficiency of the Liftbot.

## Important Note
This project is entirely theoretical and was developed for academic purposes. No data from Kewazo was used, and the project is not affiliated with or endorsed by Kewazo.

## Usage
Run the script and input the total weight, scaffold height, and scaffold length to predict the project time.


## Repository Contents
- `Multilinear_regression.py`: Python script for regression analysis.
- `project_data_20.09.csv`: Dataset used for analysis.
- `multilinear_regression_theory.pdf`: Theoretical framework and methodology.


## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone https://github.com/munycc/Multivariate-Regression-Analysis.git
   cd Multivariate-Regression-Analysis
   ```

2. **Install Dependencies**
   Install the required Python libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

3. **Run the Script**
   Execute the script to perform the regression analysis:
   ```bash
   Multilinear_regression.py
   ```



## Results
The model’s performance is evaluated using the R² score, and results are visualized to illustrate the fit of the model to the data.

