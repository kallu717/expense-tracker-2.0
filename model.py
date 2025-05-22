from sklearn.ensemble import IsolationForest

def train_anomaly_model(df):
    """
    Train a simple anomaly detector on past expense amounts.
    """
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(df[['amount']])
    return model

def is_anomaly(model, amount):
    """
    Return True if this amount is considered an anomaly (i.e. unusually large).
    """
    return model.predict([[amount]])[0] == -1
