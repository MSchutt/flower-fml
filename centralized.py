import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from pathlib import Path

from diamondmodel.loader import load_data


def mean_absolute_error(y_test, y_pred):
    return np.mean(np.abs(y_pred - y_test))

# Helper function to print scores to stdout
def calculate_and_print_scores(name, model, X_train, y_train, X_test, y_test):
    r2sq = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mean_absolute_error_test = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    r2sq_train = model.score(X_train, y_train)
    y_pred_train = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mean_absolute_error_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)

    print(f'''
        Model {name} Scores:
        *** TRAINSET ***
        R2: {r2sq_train}
        MSE: {mse_train}
        RMSE: {rmse_train}
        Mean Absolute Err: {mean_absolute_error_test}
        ****************

        *** TESTSET ***
        Model {name} Scores:
        R2: {r2sq}
        MSE: {mse}
        RMSE: {rmse}
        Mean Absolute Err: {mean_absolute_error_train}
        ****************
    ''')

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("Airbnb Data Prediction")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()


# Helper function to brute force the best parameters
def brute_force_model(X_train, y_train):
    rfr_local = RandomForestRegressor()
    # Possible values
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    rf_random = RandomizedSearchCV(estimator=rfr_local, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train)

    print(rf_random.best_estimator_)
    print(rf_random.best_score_)
    return rf_random.best_estimator_


# Centralized reference model
if __name__ == '__main__':

    pth = Path('data/diamonds.csv')
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_data(pth)

    # brute_force_model(X_train, y_train)
    print('Fitting RandomForestRegressor')
    rfr = RandomForestRegressor()
    rfr.fit(X_train, y_train)
    calculate_and_print_scores('Random Forest', rfr, X_train, y_train, X_test, y_test)
    print('Finished fitting RandomForestRegressor')

    # Linear regression
    print('Fitting LinearRegression')
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    calculate_and_print_scores('LinearRegression', lr, X_train, y_train, X_test, y_test)
    print('Finished fitting LinearRegression')


