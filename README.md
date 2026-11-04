# Dimensionless-but-not-featureless-TCL-model

by Marco Graffiedi

This repo contains the code associated with our paper "A dimensionless but not featureless machine learning generalization of a contact line evaporation model".

## Features
-  The trained XGBoost surrogate in ml_dtrfree_xgb_pi1.json and ml_dtrfree_xgb_theta_app.json, which can be interrogated using the function in tcl_predictor.py
-  The class required to solve the Mathieu model numerically in Mathieu.py
-  The function used to calculate the FC-72 properties is FC72_lib.py
-  An example of usage in TCL_model_vs_simulation.ipynb, where we compared Mathieu triple-contact-line model with the XGBoost surrogate
-  Requirements in `requirements.txt`.

## Expected results
Results should look like

<img width="571" height="431" alt="image" src="https://github.com/user-attachments/assets/444f2716-83b6-47e1-accf-79b3e912a0d8" />

<img width="571" height="431" alt="image" src="https://github.com/user-attachments/assets/b8c41dc2-93ae-4b19-afe5-f2df0a074cc7" />

## Model
The XGBoost model here reported is the one trained on the Scenario S4. The performance are here repoted as MAPE

| Scenario | Target                         | Train | Val  | Test |
| -------- | ------------------------------ | ----- | ---- | ---- |
| S4       | Π1                             | 4.3%  | 4.7% | 8.6% |
| S4       | θapp                           | 3.2%  | 4.4% | 7.6% |

## Troubleshooting
In case of issues running the code, an effective combination of packages is the following

numpy 2.5.4
scipy 1.18.1
CoolProp 7.2.0
matplotlib 3.11.2
pandas 2.3.3
xgboost 3.1.0

## References

