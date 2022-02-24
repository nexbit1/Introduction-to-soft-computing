#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np


# In[75]:


def signum(x):
    return -1 if x < 0 else 1


# In[76]:


training_input = np.array([[2, 1, 1], [0, -1, -1]])
weights = np.array([0, 1, 0])
d = np.array([-1, 1])
print('training inputs:\n',training_input)
print('initial weights:\n',weights)
print('d:\n', d)


# In[77]:


c =1
for count in range(10):
    for i in training_input:
        fnet = signum(np.dot(i, weights))
        for j in d:
            del_w = np.dot(i, j - fnet)
        weights += del_w
    print(f'ITERATION {count}:', weights)
    


# In[80]:


#FOR RANDOM WEIGHTS
weights = np.random.randint(0, 2, 3) 
#if decimal then use rand(but gives double brackets🤧, then just do weight[0]😃)
print('random initial weights:\n',weights)


# In[81]:


c =1
for count in range(10):
    for i in training_input:
        fnet = signum(np.dot(i, weights))
        for j in d:
            del_w = np.dot(i, j - fnet)
        weights += del_w
    print(f'ITERATION {count}:', weights)
    


# In[ ]:




