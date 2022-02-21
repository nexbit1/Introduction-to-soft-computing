#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[15]:


array = [(0, 0), (0, 1), (1, 0), (1, 1)]

def unipolar(f):
    if f >= 0:
        return 1
    else:
        return 0

def net(x, w, b):
    f = np.dot(x, w) + b
    y = unipolar(f)
    return y


# In[16]:


#OR LOGIC
def orgate(x):
    w = np.array([1, 1])
    b = -1
    return net(x, w, b)
for i in array:
    print(f'INPUT:[{i}] ==> OUTPUT:{orgate(i)}')


# In[17]:


#AND LOGIC
def andgate(x):
    w = np.array([1, 1])
    b = -2
    return net(x, w, b)
for i in array:
    print(f'INPUT:[{i}] ==> OUTPUT:{andgate(i)}')


# In[18]:


#NOT LOGIC
array_not = [0, 1]
def notgate(x):
    w = np.array([-1])
    b = 0
    return net(x, w, b)
for i in array_not:
    print(f'INPUT:[{i}] ==> OUTPUT:{notgate(i)}')


# In[31]:


#NAND LOGIC
def nandgate(x):
    w = np.array([-1, -1])
    b = 1
    return net(x, w, b)
for i in array:
    print(f'INPUT:[{i}] ==> OUTPUT:{nandgate(i)}')  


# In[29]:


#NOR LOGIC
def norgate(x):
    w = np.array([-1, -1])
    b = 0
    return net(x, w, b)
for i in array:
    print(f'INPUT:[{i}] ==> OUTPUT:{norgate(i)}') 


# In[ ]:




