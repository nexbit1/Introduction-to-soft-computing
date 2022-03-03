#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
get_ipython().system('{sys.executable} -m pip install -U scikit-fuzzy')


# In[4]:


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# In[5]:


# New Antecedent/Consequent objects hold universe variables and membership
# functions
quality = ctrl.Antecedent(np.arange(0, 11), 'quality')
service = ctrl.Antecedent(np.arange(0, 11), 'service')
tip = ctrl.Consequent(np.arange(0, 26), 'tip')

# Auto-membership function population is possible with .automf(3, 5, or 7)
quality.automf(3)
service.automf(3)
#tip.automf(3)
# Custom membership functions can be built interactively with a familiar,
# Pythonic API
tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])


# In[6]:


quality.view()


# In[7]:


service.view()


# In[8]:


tip.view()


# In[9]:


rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
rule2 = ctrl.Rule(service['average'], tip['medium'])
rule3 = ctrl.Rule(service['good'] | quality['good'], tip['high'])


# In[10]:


tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])


# In[11]:


tipping = ctrl.ControlSystemSimulation(tipping_ctrl)


# In[12]:


# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
tipping.input['quality'] = 9.5
tipping.input['service'] = 4.8

# Crunch the numbers
tipping.compute()


# In[13]:


print (tipping.output['tip'])
tip.view(sim=tipping)


# In[ ]:




