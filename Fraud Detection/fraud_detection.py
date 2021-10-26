# %%
'''
# Credit Card Fraud Detection

The data used for this notebooks has already had principle component analysis
(PCA) applied for confidentiality.

Dataset available at https://www.kaggle.com/mlg-ulb/creditcardfraud
'''

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('data/creditcard.csv', header=0)
df.head()

# %%
df.shape

# %%
df.dtypes

# %%
df.describe()

# %%
df.info()

# %%
fraud_percent = np.sum(df['Class']) / len(df['Class'])

# %%
fig, ax = plt.subplots()
ax.pie([1.0 - fraud_percent, fraud_percent],
       labels=['Non-Fraud', 'Fraud'],
       autopct='%1.4f%%',
       pctdistance=1.25,
       labeldistance=1.55)
plt.title('Fraud vs Non-Fraud')
plt.savefig('images/pie.png')

# %%

