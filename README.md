# Wealth Coach Lite

## Overview

Wealth Coach Lite is a Python Flask web application that provides personalized investment recommendations based on your banking transaction history and investment preferences. It uses clustering and financial analytics to infer your risk profile and suggests suitable Citi investment products.

## Features

-   **Automated Risk Profiling:**

    -   Analyzes your savings, deposits, and withdrawals using KMeans clustering.
    -   Infers your risk profile (Conservative, Balanced, Aggressive) from transaction patterns.

-   **Investment Recommendation Engine:**

    -   Matches your preferred investment types, expected returns, horizon, and liquidity needs.
    -   Suggests Citi investment products tailored to your risk profile and preferences.

-   **Interactive Web Interface:**
    -   User-friendly form to input investment preferences.
    -   Displays recommendations with detailed product info and notes on mismatches.

## Key Components

-   **Data Processing:**

    -   Loads and cleans banking transaction data from `bank.xlsx`.
    -   Aggregates monthly savings and computes financial features for risk analysis.

-   **Risk Profiling Algorithm:**

    -   Uses KMeans clustering and silhouette scoring to determine optimal risk clusters.
    -   Assigns risk profile based on savings volatility and financial behavior.

-   **Recommendation Logic:**
    -   Filters investment products by risk, type, expected return, horizon, and liquidity.
    -   Highlights mismatches in horizon and liquidity for transparency.

## Usage

1. **Setup:**

    - Ensure Python 3.x is installed.
    - Install required packages:
        ```bash
        pip install flask pandas scikit-learn openpyxl
        ```

2. **Prepare Data:**

    - Place your banking transaction file as `bank.xlsx` in the project directory.

3. **Run the Application:**

    - Start the Flask server:
        ```bash
        python app.py
        ```
    - Open your browser and go to `http://localhost:5000`.

4. **Get Recommendations:**
    - Fill out the investment preference form.
    - View personalized recommendations and inferred risk profile.

## Notes

-   The app currently supports Citi investment products as examples.
-   Ensure your transaction data is formatted correctly for best results.
-   For production use, configure security and deployment settings as needed.

Contributions to Wealth Coach Lite are welcome! Whether it's bug fixes, new investment products, or UI improvements, feel free to contribute and help enhance this project.
