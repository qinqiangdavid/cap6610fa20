import math
import matplotlib.pyplot as plt

w11=0.1
w12=0.25
w21=0.1
w22=0.7
w13=0.4
w14=1
w23=0.6
w24=0.3

x1=1
x2=1

d1=1
d2=0

lr=0.1

def phi(x):
	return 1/(1+math.exp(-x))

EPOCHS = 1000

loss=[]
for epoch in range(EPOCHS):
	n1 = w11*x1+w21*x2
	n2 = w12*x1+w22*x2

	n1o=phi(n1)
	n2o=phi(n2)

	# print("n1=",n1)
	# print("n2=",n2)

	n3 = w13*n1o+w23*n2o
	n4 = w14*n1o+w24*n2o

	y1=phi(n3)
	y2=phi(n4)

	# print("y1=",y1)
	# print("y2=",y2)
	# print("w11={},w12={},w21={},w22={}".format(w11,w12,w21,w22))
	w11=w11-lr*(((-d1+1*y1)*phi(n3)*(1-phi(n3))*w13+(-d2+1*y2)*phi(n4)*(1-phi(n4))*w14)*phi(n1)*(1-phi(n1))*x1)
	w12=w12-lr*(((-d1+1*y1)*phi(n3)*(1-phi(n3))*w23+(-d2+1*y2)*phi(n4)*(1-phi(n4))*w24)*phi(n1)*(1-phi(n1))*x1)
	w21=w21-lr*(((-d1+1*y1)*phi(n3)*(1-phi(n3))*w13+(-d2+1*y2)*phi(n4)*(1-phi(n4))*w14)*phi(n1)*(1-phi(n1))*x2)
	w22=w22-lr*(((-d1+1*y1)*phi(n3)*(1-phi(n3))*w23+(-d2+1*y2)*phi(n4)*(1-phi(n4))*w24)*phi(n1)*(1-phi(n1))*x2)
	print("w11={},w12={},w21={},w22={}".format(w11,w12,w21,w22))


	# print("w13={},w14={},w23={},w24={}".format(w13,w14,w23,w24))
	w13=w13-lr*((-d1+1*y1)*phi(n3)*(1-phi(n3))*n1o)
	w14=w14-lr*((-d2+1*y2)*phi(n4)*(1-phi(n4))*n1o)
	w23=w23-lr*((-d1+1*y1)*phi(n3)*(1-phi(n3))*n2o)
	w24=w24-lr*((-d2+1*y2)*phi(n4)*(1-phi(n4))*n2o)
	print("w13={},w14={},w23={},w24={}".format(w13,w14,w23,w24))

	J = 1*((d1-y1)**2+(d2-y2)**2)
	loss.append(J)
	# if(epoch%100==0):
		# print("loss={}".format(J))


x=range(EPOCHS)
plt.plot(x,loss)
plt.show()