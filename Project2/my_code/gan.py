import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import idx2numpy
import random
import torchvision

def show_images(images): # 定义画图工具
    # plt.figure()
    # print(images.shape[0])
    for i in range(images.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(images[i-1].reshape(28,28))
        plt.axis('off')
    # plt.show()

file = '../MNIST'
train_image = idx2numpy.convert_from_file(file+'/train-images.idx3-ubyte')
train_label = idx2numpy.convert_from_file(file+'/train-labels.idx1-ubyte')
test_image = idx2numpy.convert_from_file(file+'/t10k-images.idx3-ubyte')
test_label = idx2numpy.convert_from_file(file+'/t10k-labels.idx1-ubyte')


cuda = False
if(torch.cuda.is_available()):
    cuda = True

print(cuda)

fives=[]

for i in range(train_image.shape[0]):
    if train_label[i]==4:
        fives.append(train_image[i])
fives=np.array(fives)
# print(fives.shape)

fives=np.reshape(fives,(fives.shape[0],28*28))

# print(fives.shape)

# Hyper Parameters
BATCH_SIZE = 512
EPOCHS = 200
BATCHS = int(fives.shape[0]/BATCH_SIZE)
# EPOCHS = 500
LR_G = 0.0001      # learning rate for generator
LR_D = 0.0001       # learning rate for discriminator
N_IDEAS = fives.shape[0]         # think of this as number of ideas for generating an art work(Generator)
IMAGE_SIZE = 28*28 # it could be total point G can drew in the canvas
# PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

G = nn.Sequential(                  # Generator
    nn.Linear(N_IDEAS, 128),        # random ideas (could from normal distribution)
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 256),        # random ideas (could from normal distribution)
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(256, 512),        # random ideas (could from normal distribution)
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(512, IMAGE_SIZE) # making a painting from these random ideas
    # nn.Tanh()
)

D = nn.Sequential(                  # Discriminator
    nn.Linear(IMAGE_SIZE, 512),     # receive art work either from the famous artist or a newbie like G
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(512, 256),        # random ideas (could from normal distribution)
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(256, 1),
    nn.Sigmoid(),                   # tell the probability that the art work is made by artist
)

if cuda:
    G=G.cuda()
    D=D.cuda()
# loss function


opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()    # something about continuous plotting

D_loss_history = []
G_loss_history = []


# plt.show()

print(EPOCHS)
for epoch in range(EPOCHS):
    for step in range(BATCHS):
        # artist_paintings = fives[random.randint(0,fives.shape[0]-1)]          # real painting from artist
        artist_paintings = fives[step*BATCH_SIZE:step*BATCH_SIZE+BATCH_SIZE]
        artist_paintings = torch.Tensor(artist_paintings)
        # print(artist_paintings.shape)
        G_ideas = torch.randn(BATCH_SIZE, N_IDEAS) # random ideas
        if cuda:
            G_ideas = G_ideas.cuda()
        G_paintings = G(G_ideas)                    # fake painting from G (random ideas)

        if cuda:
            artist_paintings = artist_paintings.cuda()
            G_paintings = G_paintings.cuda()

        prob_artist0 = D(artist_paintings)         # D try to increase this prob
        prob_artist1 = D(G_paintings)              # D try to reduce this prob

        D_loss = - torch.mean(torch.log(0.-prob_artist0) + torch.log(1. - prob_artist1))
        G_loss = torch.mean(torch.log(1. - prob_artist1))

        D_loss_history.append(D_loss)
        G_loss_history.append(G_loss)
        

        opt_D.zero_grad()
        D_loss.backward(retain_graph=True)    # reusing computational graph
        opt_D.step()

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        # plt.cla()
        
        # if(step%10==0):
            # print("batch/batchs:{}/{}".format(step,BATCHS))
    mean = G_paintings.data.float().mean()
    std = G_paintings.data.float().std()
    G_paintings.data = (G_paintings.data-mean)/std
    imgs = G_paintings.cpu().data.numpy()
        # show_images(imgs[0:16])
    # print(mean)
    # threshold = ((G_paintings.data.float().max()-G_paintings.data.float().min())*0.5+G_paintings.data.float().min()).item()
    # threshold = (imgs.max()-imgs.min())*0.5+imgs.min()
    # imgs[imgs >= threshold] = 1
    # imgs[imgs < threshold] = 0
    plt.imshow(imgs[20].reshape(28,28))
    # plt.draw();
    plt.pause(0.01)
    
    print("epoch：",epoch)


# img = G_paintings.data.numpy()[0].reshape(28,28)
            # print(img)
# plt.imshow(img)
# plt.show()
plt.ioff()
plt.show()