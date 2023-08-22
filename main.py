import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


# Read the CSV file
df = pd.read_csv('fast_food.csv')

# Filter rows for "McDonald’s" or "KFC" companies
df_filtered = df[df['Company'].isin(["McDonald’s", "KFC"])]

# Remove rows with missing values
df_filtered = df_filtered.dropna(subset=['Calories', 'Total Fat\n(g)'])


# Split the data
train, valid, test = np.split(df_filtered.sample(frac=1), [int(0.6*len(df_filtered)), int(0.8*len(df_filtered))])

def scale_dataset(dataframe, oversample=False):
    X = dataframe[['Calories', 'Total Fat\n(g)']].values
    y = dataframe['Company'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    return X, y

# Scale and split the datasets
X_train, y_train = scale_dataset(train, oversample=True)
X_valid, y_valid = scale_dataset(valid, oversample=False)
X_test, y_test = scale_dataset(test, oversample=False)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

print(classification_report(y_test, y_pred))