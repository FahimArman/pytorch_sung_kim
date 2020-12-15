import torch
from torch import nn
from torch import tensor

from torch import sigmoid
import torch.nn.functional as F
import torch.optim as optim

x_data = tensor([[1.0],[2.0],[3.0],[4.0]])
y_data = tensor([[0.0],[0.0],[1.0],[1.0]])

class Model(torch.nn.Module):
	def __init__(self):
		super(Model,self).__init__()
		self.linear=torch.nn.Linear(1,1)
	
	def forward(self,x):
		y_pred = sigmoid(self.linear(x))
		return y_pred

model = Model()

criterion = torch.nn.BCELoss(reduction = 'mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(500):
	y_pred = model(x_data)
	loss = criterion(y_pred,y_data)
	
	print("Epoch: ",epoch+1," Loss: ", loss.item())
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()


print(f'\nLet\'s predict the hours need to score above 50%\n{"=" * 50}')
houro_var = model(tensor([[1.0]]))
print(f'Prediction after 1 hour of training: {houro_var.item():.4f} | Above 50%: {houro_var.item() > 0.5}')
hours_var = model(tensor([[7.0]]))
print(f'Prediction after 7 hours of training: {hours_var.item():.4f} | Above 50%: { hours_var.item() > 0.5}')



