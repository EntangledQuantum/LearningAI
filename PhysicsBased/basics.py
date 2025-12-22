import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using CUDA:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available, using CPU.")

# Generate data
N = 10000
X = np.random.random(N).astype(np.float32).reshape(-1, 1)

# Generation of Y-Data
sign = (- np.ones((N,))).astype(np.float32) ** np.random.randint(2, size=N)
Y = (np.sqrt(X.flatten()) * sign).reshape(-1, 1).astype(np.float32)

# Convert to PyTorch tensors and send to device
X_tensor = torch.tensor(X).to(device)
Y_tensor = torch.tensor(Y).to(device)

criterion = nn.MSELoss()

# X-Data
# X = X , we can directly re-use the X from above, nothing has changed...

# P maps Y back to X, simply by computing a square, as y is a TF tensor input, the square operation **2 will be differentiable
def P(y):
    return torch.square(y)

# Define custom loss function using the "physics" operator P
def loss_function(x_true, y_pred):
    return criterion(x_true, P(y_pred))