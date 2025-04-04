import torch
import numpy as np

np.random.seed(42)
x = np.linspace(-10, 10, 100)
y = 2 * x + 1 + np.random.normal(0, 1, 100)

X = torch.FloatTensor(x.reshape(-1, 1))
Y = torch.FloatTensor(y.reshape(-1, 1))

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(X)
    loss = criterion(outputs, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

slope = model.linear.weight.item()
intercept = model.linear.bias.item()

print(f"y = {slope:.4f}x + {intercept:.4f}")
print(f"y = 2x + 1")