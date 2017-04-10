import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
#import matplotlib.pyplot as plt

df = pd.read_csv('origin_output.csv')
dg = df.groupby('author')
for k, g in dg:
    model = ols('bool_increase ~ rise+drop', g)
    results = model.fit()
    print(results.summary())
#print(df.describe())

