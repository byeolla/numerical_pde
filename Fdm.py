#!/usr/bin/env python
# coding: utf-8

# In[142]:


import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
#####choco pie###
pi = torch.Tensor([math.pi])
#####lets mesh#####
M,N=20000,20000
x=torch.linspace(-3,3,M)
t=torch.linspace(0,2,N)
X,T = torch.meshgrid(x,t)
##########Initial function####
def Initial_function(x):
    result=torch.where(x>-0.5,torch.cos(pi*x)**2,x*0)
    result=torch.where(x<0.5,result,x*0)
    return result

def True_solution(x,t):
    return Initial_function(x-t)
#########lets make  mesh function####

F=torch.zeros([M,N])

MM=6*(M**(-1))
NN=2*(N**(-1))
for i in range(N):
    if i==0:
        F[:,0]=Initial_function(X[:,0])
    else:
        F[1:M,i]=F[1:M,i-1]-(F[1:M,i-1]-F[0:M-1,i-1])*(MM**(-1))*NN
        


# In[143]:


#####plot
X11 = torch.linspace(-3,3,steps=M)
fig = plt.figure(figsize=(10,10))
#y=torch.tensor(0.25).type(torch.float64).to(device)
X11=X11.unsqueeze(1)

Z_p=True_solution(X11,1)#.numpy()
#Z_t=Initial_function(X[:,0])
Z_t=F[:,10000-1]

plt.plot(X11.cpu(), Z_t, '-x',label="Fdm")
plt.plot(X11.cpu(), Z_p, label="Exact_solution")
plt.legend()


# In[ ]:




