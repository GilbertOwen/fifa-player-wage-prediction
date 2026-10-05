# FIFA 21 Player Wage Prediction

Predicting a football player's weekly wage (in euros) from their FIFA 21 stats, with LightGBM.

The whole process is in [`ml.ipynb`](ml.ipynb), from choosing the target to the final model, including the mistakes along the way.

## Result

| Model | Test MAE (€ per week) |
|---|---|
| Baseline, always guess the median (3,000) | 7,754 |
| LightGBM | 3,974 |
| LightGBM + log(Wage) | 3,921 |
| LightGBM + club median wage | 1,533 |
| LightGBM + club median wage + log(Wage) | **1,485** |

The honest number for the final model is **€1,527 per week**, from 5-fold cross-validation on the training set. The test set was used to choose between the options above, so 1,485 is a bit flattering.

## What's in the notebook

1. Load the data
2. Define the problem: predict Wage
3. Feature selection and leakage checks (ratio tests for Value, POT, Release Clause), plus feature engineering (contract_length, years_at_club, on_loan)
4. Train/test split
5. Baseline
6. Six algorithms compared: Linear Regression, Random Forest, Gradient Boosting, SVR, XGBoost, LightGBM
7. Evaluation: overfitting check, feature importance (split vs gain)
8. Error diagnosis: error by wage band and the direction of the error
9. Experiment: training on log(Wage)
10. Experiment: dropping Value, checked with K-fold
11. Experiment: bringing Team back as target encoding (club median wage), honest vs leaky K-fold
12. Final model and conclusion

## What I learned

- Six algorithms all stopped around 4,000, so the algorithm was never the problem. The missing piece was the club: without it, the model can't tell apart two players with the same rating at very different clubs.
- My first diagnosis was wrong. In section 8 I blamed the skewed target, and log only helped a little. Measuring the error in percent also showed that the euro table had been hiding the worst band (the cheapest players, 209% error).
- Target encoding leaks if you're careless. Building the club medians before K-fold split the data made the score look €154 better than it really was (1,442 vs 1,596).
- One split can lie. Dropping Value looked 16 euro better on one split and was worse in 4 of 5 K-fold rounds.

## Limitations

- The wages are EA's numbers from the game, weekly, in euros, not real contracts.
- The model needs Value and Release Clause, so it can't price a player who has no market value yet.
- Top earners are under-predicted. Messi (€560,000) is predicted at about €306,000: nobody in the training set earned more than €350,000, and tree models can't predict past what they've seen.
- Free agents (237 players with no club and a wage of 0) are left out.

## Run it

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then open `ml.ipynb` and run all cells.

`fifa21_cleaned.csv` has 18,978 players. It was cleaned from raw FIFA 21 player data scraped from sofifa.com, in a separate data cleaning project.

A web app built on this model lives in a separate repository, [fifa-wage-prediction-app](https://github.com/GilbertOwen/fifa-wage-prediction-app). Try it live: [fifa-wage-prediction.streamlit.app](https://fifa-wage-prediction.streamlit.app/)
