import numpy as np
import math
import torch 
import torch.nn as nn
from torch.autograd import Variable

myNet = nn.Sequential(
	nn.Linear(6, 6),
	# nn.Sigmoid(),
	# nn.Linear(10, 6),
	# nn.Sigmoid(),
)

EPOCHS=100
learning_rate=0.01

for epoch in range(EPOCHS):
	input = torch.rand(6)
	input = input / torch.sum(input)
	input=Variable(input,requires_grad=True)
	opt = torch.optim.Adam(myNet.parameters(),lr=learning_rate)

	psum=torch.Tensor([0])
	pmsum = psum
	mysum=psum
	for i in range(6):
		psum = psum+(i+1)*input[i]
		pmsum = pmsum+(torch.pow(input[i],input[i]))

	for j in range(6):
		i=j+1
		pmul = torch.Tensor([1])
		for k in range(i):
			pmul = pmul*input[k]

		mysum=mysum+(pmul-torch.pow(pmsum,i/psum))

	loss = mysum
	# opt.zero_grad()
	loss.backward()
	opt.step()
	print(input)
	print(loss)
