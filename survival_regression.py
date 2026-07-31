import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

# 1. Load example survival dataset (Rossi parole data)
df = load_rossi()

# 2. Fit Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df, duration_col='week', event_col='arrest')

# 3. Print summary
print(cph.summary)

# 4. Predict survival function for first 5 individuals
survival_funcs = cph.predict_survival_function(df.iloc[:5])
print(survival_funcs.head())
