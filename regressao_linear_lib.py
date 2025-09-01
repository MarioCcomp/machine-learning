import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import seaborn as sns
import math

base = pd.read_csv("mt_cars.csv")

base = base.drop(['Unnamed: 0'], axis=1)

print(base.head())

corr = base.corr()
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')


column_pairs = [('mpg', 'cyl'), ('mpg', 'disp'), ('mpg', 'hp'), ('mpg', 'wt'), ('mpg', 'drat'), ('mpg', 'vs')]

n_plots = len(column_pairs)
ncols = 2
nrows = math.ceil(n_plots / ncols)
gig, axes = plt.subplots(nrows=nrows, ncols = ncols, figsize=(12,4 * n_plots))
gig.suptitle("Análise de Correlação: MPG e outras variáveis", fontsize=16)

axes = axes.flatten()


for i, pair in enumerate(column_pairs):
    x_col, y_col = pair
    sns.scatterplot(x=x_col, y=y_col, data=base, ax=axes[i])
    axes[i].set_title(f'{x_col} vs {y_col}')
    # axes[i].set_xlim(0, 25) 

plt.tight_layout()
# plt.show()

# aic 156.6 bic 162.5
# modelo = sm.ols(formula='mpg ~ wt + disp + hp', data=base)

# aic 165.1 bic 169.5
# modelo = sm.ols(formula='mpg ~ cyl + disp', data=base)

# aic 179.1 bic 183.5
modelo = sm.ols(formula='mpg ~ drat + vs', data=base)


modelo = modelo.fit()
print(modelo.summary())

plt.figure(figsize=(8,5))
residuos = modelo.resid
plt.hist(residuos, bins=20)
plt.xlabel("Residuos")
plt.ylabel("Frequencia")
plt.title("Histograma de residuos")
plt.show()

stats.probplot(residuos, dist="norm", plot=plt)
plt.title("Q-Q Plot de Residuos")
plt.show()

stat, pval = stats.shapiro(residuos)
# hipotese nula - os dados estao normalmente distribuidos
#  p <= 0.05 rejeito a hipotese nula (ou seja, nao estao normalmente distribuidos)
# p > 0.05 não é possivel rejeitar a hipotese nula

# primeiro 0.033 < 0.05
# segundo 0.085 > 0.05
# terceiro 0.822 > 0.05
print(f"Shapiro-Wilk statistica: {stat:.3f}, p-value: {pval:.3f}")
