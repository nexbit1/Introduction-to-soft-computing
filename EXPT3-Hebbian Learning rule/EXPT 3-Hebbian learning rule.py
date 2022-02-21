#!/usr/bin/env python
# coding: utf-8

# In[93]:


import numpy as np


# In[94]:


def signum(x):
    if x >= 0:
        return 1
    else:
        return -1


# In[95]:


training_input = np.array([[1,-2,1.5,0],[1,-0.5,-2,-1.5],[0,1,-1,1.5]])
weights = np.array([1,-1,0,0.5])
print('Training Inputs:\n', training_input)
print('Initial weights:\n', weights)


# In[96]:


c = 1
for count in range(0,10): #for 10 iteration, 1 iteration weight update 3 times(3 inputs😃)
    for i in training_input:
        fnet = signum(np.dot(i,weights))
        del_w = np.dot(i, c*fnet)
        weights += del_w
    print(f'ITERATION {count}:', weights)
    


# In[107]:


#randomised weights
weights = np.random.rand(1, 4).round(1)
print('Random initial weight:\n', weights)
c = 1
for count in range(0,10): #for 10 iteration, 1 iteration weight update 3 times(3 inputs😃)
    for i in training_input:
        fnet = signum(np.dot(i,weights[0])) #[[]]🤔
        del_w = np.dot(i, c*fnet)
        weights += del_w
    print(f'ITERATION {count}:', weights)


# In[ ]:




