import pymc3 as pm
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. Bayesian Linear Regression with PyMC3
with pm.Model() as model:
    # Priors for coefficients
    beta = pm.Normal('beta', mu=0, sigma=1, shape=X_train_scaled.shape[1])
    intercept = pm.Normal('intercept', mu=0, sigma=1)
    
    # Expected value
    mu = pm.math.dot(X_train_scaled, beta) + intercept
    
    # Likelihood
    sigma = pm.HalfNormal('sigma', sigma=1)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train)
    
    # Sample posterior
    trace = pm.sample(1000, tune=1000, cores=1, random_seed=42)

# 5. Inspect posterior means
print(pm.summary(trace, var_names=['beta', 'intercept', 'sigma']))
