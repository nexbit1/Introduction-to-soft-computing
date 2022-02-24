#!/usr/bin/env python
# coding: utf-8

# In[160]:


import numpy as np


# In[161]:


def signum(x):
    return -1 if x < 0 else 1


# In[162]:


training_input = np.array([[2, 1, 1], [0, -1, -1]])
weights = np.array([0, 1, 0])
d = np.array([-1, 1])
print('training inputs:\n',training_input)
print('initial weights:\n',weights)
print('d:\n', d)


# In[163]:


c =1
for count in range(10):
    for j, i in enumerate(training_input): #enumerate j==>index, i==>item in iterable
        fnet = signum(np.dot(i, weights))
        del_w = np.dot(i, d[j] - fnet)
        weights += del_w
    print(f'ITERATION {count}:', weights)
    


# In[164]:


#FOR RANDOM WEIGHTS
weights = np.random.randint(0, 2, 3) 
#if decimal then use rand(but gives double brackets🤧, then just do weight[0]😃)
print('random initial weights:\n',weights)


# In[165]:


c =1
for count in range(10):
    for j, i in enumerate(training_input):
        fnet = signum(np.dot(i, weights))
        del_w = c * np.dot(i, d[j] - fnet)
        weights += del_w
    print(f'ITERATION {count}:', weights)
    


# In[ ]:





# In[ ]:




