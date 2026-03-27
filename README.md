# MLB WAR Prediction Model

A machine learning pipeline that predicts next-season WAR (Wins Above Replacement) 
for MLB batters using 20+ years of historical batting data. Built with Ridge Regression, 
sequential forward feature selection, and a rolling backtesting framework.

## Results

- **Average MSE: 2.20 WAR** across all seasons (2008–2024)
- **Average RMSE: 1.48 WAR**
- Top 20 predictors identified from 344 engineered features
- 2020 COVID season produced highest error (3.03 MSE) due to 60-game sample size

## Demo

<img width="1044" height="781" alt="image" src="https://github.com/user-attachments/assets/f3f0c378-ee85-435a-8abe-79940c6a9e2b" />

<img width="1564" height="523" alt="image" src="https://github.com/user-attachments/assets/fc54849f-a9ee-4841-a107-c309bc6a619f" />


## How It Works

1. Pull 20+ seasons of MLB batting data (2004–2025) using pybaseball
2. Engineer 344 features including plate discipline, power metrics, 
   contact rates, baserunning, age curves, and Statcast metrics
3. Align each season's stats with the following season's WAR as the target variable
4. Run sequential forward feature selection to identify the top 20 predictors
5. Validate using a rolling backtest and train only on past seasons, 
   predict each future season independently
6. Analyze over- and under-performers across player histories

## Top 20 Predictors

Age, IBB, BU, BABIP, Pos, RAR, Contact%, UBR, Med%, Zone%, 
FRM, Oppo%+, Hard%, maxEV, IBB_rate, HR_FB_proxy, 
BABIP_proxy, SB_rate, AB_per_game, PA_per_game

## Key Findings

- Age and contact metrics are the strongest predictors of next-season WAR
- Statcast metrics (Hard%, maxEV, Zone%) add signal for post-2015 seasons
- Breakout seasons (Aaron Judge 2022, Mike Trout 2015) are systematically 
  underestimated as the performance spikes are inherently unpredictable
- Injury-driven collapses (Jason Bay, Ronald Acuna Jr.) represent the 
  largest prediction errors

## Tech Stack

- **pybaseball** : MLB data ingestion
- **pandas / NumPy** : data processing and feature engineering
- **scikit-learn** : Ridge Regression, SequentialFeatureSelector, cross-validation
- **matplotlib / seaborn** : visualization
- **Jupyter Notebook** : development environment

## Setup

1. Clone the repo:
```
   git clone https://github.com/gabezhu23/mlb-war-model
   cd mlb-war-model
```

2. Install dependencies:
```
   pip install -r requirements.txt
```

3. Run the notebook:
```
   jupyter notebook mlb_war_prediction.ipynb
```
   Or open directly in VS Code.

## What I learned

- How to build a time-series ML pipeline that avoids data leakage 
  by training only on past data at each validation step
- How sequential forward feature selection identifies the most 
  predictive features from a high-dimensional dataset
- How Ridge Regression handles multicollinearity across 344 
  correlated batting statistics
- Why WAR prediction is fundamentally limited by injury and 
  breakout variance that no model can anticipate
