# %%
'''
# Credit Card Fraud Detection

The data used for this notebooks has already had principle component analysis
(PCA) applied for confidentiality. A neural network is trained to
identify credit card fraud.

Dataset available at https://www.kaggle.com/mlg-ulb/creditcardfraud
'''

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
# plot the amound of fraud vs non-fraud cases
fig, ax = plt.subplots()
ax.pie([1.0 - fraud_percent, fraud_percent],
       labels=['Non-Fraud', 'Fraud'],
       autopct='%1.4f%%',
       pctdistance=1.25,
       labeldistance=1.55)
plt.title('Fraud vs Non-Fraud')
plt.savefig('images/pie.png')

# %%
# split test and training data
X = df.drop(columns='Class')
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1984)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
# create neural network
lr = 2e-3
model = Sequential([
    Dense(10, input_dim=30),
    LeakyReLU(alpha=lr),
    Dropout(0.2),
    Dense(10),
    LeakyReLU(alpha=lr),
    Dense(10),
    LeakyReLU(alpha=lr),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
    ])
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# %%
# train neural network
epochs = 20
history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
fig, ax = plt.subplots()
ax.plot(range(1, epochs+1), history.history['val_loss'], label='val_loss')
ax.plot(range(1, epochs+1), history.history['loss'], label='loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.savefig('images/loss.png')

# %%
loss, acc = model.evaluate(X_test, y_test)
print('Model Test Accuracy: ', acc)
