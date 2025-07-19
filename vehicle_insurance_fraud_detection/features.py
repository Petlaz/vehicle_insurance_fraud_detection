from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_features_targets(df):
    X = df.drop("FraudFound", axis=1)
    y = df["FraudFound"]
    return X, y

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


