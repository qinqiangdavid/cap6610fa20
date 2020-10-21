import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

NODE_NUM = 10
EPOCHS = 1000
POINT_NUM = 100
learning_rate = 0.05

a=0.5
b=0.6
r=0.4

cuda = False
# if(torch.cuda.is_available()):
    # cuda = True

myNet = nn.Sequential(
    nn.Linear(2, NODE_NUM),
    nn.Sigmoid(),
    nn.Linear(NODE_NUM, 1),
    nn.Sigmoid(),
)

# my_net = myNet()

def in_circle(x,y,a=a,b=b,r=r):
    if np.power(x-a,2)+np.power(y-b,2)<np.power(r,2):
        return True
    else:
    	return False

def calcgt(input=input,point_num=POINT_NUM):
    x = torch.Tensor([])
    for i in range(point_num):
        if in_circle(x=input[i, 0].cpu().data.numpy(),y=input[i, 1].cpu().data.numpy())==True:
            x = torch.cat((x,torch.Tensor([1])), 0)
        else:
            x = torch.cat((x,torch.Tensor([0])), 0)
        # print(i, x.shape)
    return x


def train(net,a=a,b=b,r=r):
    accuracy = []
    if cuda:
        net = net.cuda()
    opt = torch.optim.Adam(net.parameters(),lr=learning_rate)
    # loss = nn.BCEWithLogitsLoss()
    loss = nn.MSELoss()
    for epoch in range(EPOCHS):
        input = torch.rand(POINT_NUM, 2)

        if cuda:
        	input = input.cuda()

        predict = net(input)
        predict = predict.squeeze(-1) 
        labels = calcgt(input)

        if cuda:
        	predict = predict.cuda()
        	labels = labels.cuda()
   
        net_loss = loss(predict, labels)
        opt.zero_grad()
        net_loss.backward(retain_graph=True)
        opt.step()
        accuracy.append(1-net_loss.item())
        if(epoch%100==0):
            print("Epoch[{}/{}]:loss={}".format(epoch, EPOCHS, net_loss.item()))
    input1 = torch.rand(1000,2)

    predict = net(input1)
    predict = predict.squeeze(-1) 
    labels = calcgt(input1,point_num=1000)

    hyperplane = []
    for i in range(1000):
    	if torch.abs(predict[i]-labels[i])>0.25:
    		hyperplane.append(input1[i].cpu().numpy().tolist())
    hyperplane = np.array(hyperplane)
    print(hyperplane.shape)
    plt.plot(hyperplane[:,0],hyperplane[:,1],'g+')
    return net, accuracy

# 
def test(network,input,point_num=POINT_NUM):
    predict = network(input)
    in_circle = []
    out_circle = []
    predict = predict.squeeze(-1) 
    # print(predict)
    for i in range(point_num):
    	# print(predict[i])
    	# print(input[i].numpy())
    	if predict[i].data>predict.mean().data:
    		in_circle.append(input[i].cpu().numpy().tolist())
    	else:
    		out_circle.append(input[i].cpu().numpy().tolist())
    return np.array(in_circle),np.array(out_circle)



net, loss = train(net=myNet)
# plt.plot(np.array(range(0,EPOCHS)),loss)
# plt.
# for name, param in net.named_parameters():
    # if param.requires_grad:
        # print (name, param.data)
input = torch.rand(POINT_NUM, 2)
if cuda:
	input = input.cuda()
x, y = test(net,input)
# plt.xlabel('iterations')
# plt.ylabel('accuracy')

# print(x.shape)
# print(y.shape)
# plt.figure()
plt.plot(x[:,0],x[:,1],'bo')
plt.plot(y[:,0],y[:,1],'ro')



plt.show()