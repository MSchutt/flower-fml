from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z']


def transform_data(df: pd.DataFrame):
    # Extract the relevant data
    df = df.astype({
        'carat': 'float64',
        'cut': 'category',
        'color': 'category',
        'clarity': 'category',
        'depth': 'float64',
        'table': 'float64',
        'price': 'float64',
        'x': 'float64',
        'y': 'float64',
        'z': 'float64',
    })

    # Encode the categorical vals
    categorical_vals = ['cut', 'color', 'clarity']
    # Label Encoders
    for categorical in categorical_vals:
        encoder = LabelEncoder()
        encoder.fit(df[categorical])
        df[categorical] = encoder.transform(df[categorical])
    # StandardScale all columns
    StandardScaler().fit_transform(df)

    X = df.drop('price', axis=1)
    y = df['price']
    return X, y

def load_data(file_path: Path = None, random_state=42, test_size=0.3):
    if file_path is None:
        file_path = Path('./data/diamonds.csv')
    data = pd.read_csv(file_path, usecols=columns)

    X, y = transform_data(data)

    # Train, Valid, Test Split
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, test_size=0.3, random_state=random_state)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
