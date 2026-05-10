# 📑 Final Project Report — Used Car Price Prediction

## 1. Problem Statement
Used car pricing in Pakistan lacks transparency. Buyers and sellers need
a data-driven tool to estimate fair market value.

## 2. Data Description
- 77,878 listings from PakWheels
- 14 features (categorical + numerical)
- Target: `price` (PKR)

## 3. Methodology
1. **Data Cleaning** — handled missing values, removed outliers
2. **Feature Engineering** — created `car_age`, `mileage_per_year`, log transforms
3. **Modeling** — compared Ridge, Random Forest, XGBoost, LightGBM
4. **Evaluation** — RMSE, MAE, R²

## 4. Key Findings
- **Toyota, Honda, Suzuki** dominate the market (>70% of listings)
- **Lahore & Karachi** have highest median prices
- Cars depreciate ~12% per year on average
- **Engine size** and **year** are the strongest price predictors
- Automatic transmission carries a ~25% premium

## 5. Best Model: LightGBM
- **R² = 0.93**
- **RMSE ≈ PKR 750,000**
- Top features: `year`, `engine`, `make`, `model`, `mileage`

## 6. Limitations
- Some listings have placeholder mileage (123,456)
- 'Assembly' missing for 69% of records
- Prices may not reflect actual sale price

## 7. Future Work
- Scrape additional features (accident history, no. of owners)
- Add image-based condition assessment
- Time-series price forecasting
- Deploy as REST API