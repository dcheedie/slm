import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
import statsmodels.regression.linear_model as lm
from statsmodels.stats.anova import anova_lm

from scipy import stats
from sklearn import linear_model

# Calculate sum of squares for a numpy array or pandas dataframe
def calc_SS(obs):
    mean = np.mean(obs)
    mean_arr = np.full_like(obs, mean, dtype=np.float)
    diff = obs - mean_arr
    diff_sq = diff**2
    return np.sum(diff_sq)

'''
Create simple linear model, fit, and get results for x, y


'''
def run_slm(df, xcolname, ycolname):
	model = smf.ols('{} ~ {}'.format(ycolname, xcolname), data=df)
	results = model.fit()
	summary = results.summary()
	anova = sm.stats.anova_lm()
	return dict({"model": model,
				 "results": results,
				 "summary": summary,
				 "anova": anova})

def pred_analysis(results):
	results.get_prediction()



# Basic scatter plot of x and y
# def plot_xy(x, y, title):
# 	mpl.style.use('seaborn-muted')
# 	ax.scatter(x, y, fc='w')
# 	ax.plot(x, y_fit)
# 	ax.set_title("{}".format(title)

# Plot t distribution for hypothesis testing