import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. For demo, create synthetic "time-to-event" data
# Let's pretend target y is survival time, and censoring indicator is random
np.random.seed(42)
event_observed = np.random.binomial(1, 0.8, size=len(y))  # 80% observed events

# 3. Split data
X_train, X_test, y_train, y_test, e_train, e_test = train_test_split(
    X, y, event_observed, test_size=0.3, random_state=42
)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 5. Prepare dataframe for lifelines
df_train = pd.DataFrame(X_train_scaled, columns=california.feature_names)
df_train['time'] = y_train
df_train['event'] = e_train

# 6. Fit Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df_train, duration_col='time', event_col='event')

# 7. Inspect coefficients
cph.print_summary()
