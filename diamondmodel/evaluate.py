import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error

# Helper function that evaluates a model
def evaluate_model(model, X_test, y_test, device):
    testing_X = torch.from_numpy(X_test.values).float()
    testing_y = torch.from_numpy(y_test.values).float()

    model.eval()
    with torch.no_grad():
        predictions = model(testing_X.to(device))
        predictions = predictions.cpu().numpy()
        # calculate statistics
        r2 = r2_score(testing_y, predictions)
        mse = mean_squared_error(testing_y, predictions)
        rmse = np.sqrt(mse)

        print(f'R2 Score: {r2}')
        print(f'MSE: {mse}')
        print(f'RMSE: {rmse}')
    return r2, mse, rmse