import torch
from torch import nn
from torch import tensor

x_data = tensor([[1.0],[2.0],[3.0]])
y_data = tensor([[2.0],[4.0],[6.0]])

class Model(torch.nn.Module):
	def __init__(self):
		super(Model,self).__init__()
		self.linear = torch.nn.Linear(1,1)
	
	def forward(self,x):
		y_pred = self.linear(x)
		return y_pred

model = Model()

criterion = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)


for epoch in range(500):
	y_pred = model(x_data)
	loss = criterion(y_pred,y_data)
	print("epoch",epoch,"loss: ",loss.item())
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

hour_var = tensor([[4.0]])
print("after training", 4, model(hour_var).data[0][0].item())

