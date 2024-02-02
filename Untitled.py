#!/usr/bin/env python
# coding: utf-8

# In[1]:


age = [30,40,30,49,22,35,22,46,29,25,39]
age


# In[2]:


import pandas as pd
df = pd.read_csv("country_data.csv")
df


# In[3]:


import pandas as pd
df = pd.read_csv("country_data.csv")
df


# In[4]:


import pandas as pd
df = pd.read_csv("country_data.csv")
df


# In[5]:


import pandas as pd
df = pd.read_csv("country_data.csv")
df


# In[6]:




age=[39, 25, 29, 46, 22, 35, 22, 49, 30, 40, 30]

mean = sum(age)/len(age)

print(mean)


# In[7]:


sq_diff_list = []

for i in range(len(age)):
    sq_diff = (age[i] - mean)**2
    sq_diff_list.append(sq_diff)
    print(sq_diff_list)


# In[8]:


sq_diff_list = []

for i in range(len(age)):
    sq_diff = (age[i] - mean)**2
    sq_diff_list.append(sq_diff)
    print(sq_diff)


# In[9]:


mean_sq_diff = sum(sq_diff_list)/len(sq_diff_list)
print(mean_sq_diff**0.5)


# In[10]:


import numpy
age=[39, 25, 29, 46, 22, 35, 22, 49, 30, 40, 30]
print(numpy.std(age))


# In[11]:


import numpy as np
hours = [29, 9, 10, 38, 16, 26, 50, 10, 30, 33, 43, 2, 39, 15, 44, 29, 41, 15, 24, 50]
results = [65, 7, 8, 76, 23, 56, 100, 3, 74, 48, 73, 0, 62, 37, 74, 40, 90, 42, 58, 100]
x = np.asarray(hours)
y = np.asarray(results)
sum_x = np.sum(x)
sum_x_2 = np.sum(np.square(x))
sum_y = np.sum(y)
sum_y_2 = np.sum(np.square(y))
sum_xy = np.sum(x*y)
print(f"sum_x = {sum_x}")
print(f"sum_x_2 = {sum_x_2}")
print(f"sum_y = {sum_y}")
print(f"sum_y_2 = {sum_y_2}")
print(f"sum_xy = {sum_xy}")
n = len(x)
print(f"n = {n}")
top = n*sum_xy - sum_x*sum_y
print(f"top = {top}")
bot_a = np.sqrt(n*sum_x_2 - np.square(sum_x))
bot_b = np.sqrt(n*sum_y_2 - np.square(sum_y))
bot = bot_a*bot_b
print(f"bot = {bot}")
R_2 = np.square(top/bot)
print(f"R_2 = {R_2}")


# In[12]:


import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Data
hours = [29, 9, 10, 38, 16, 26, 50, 10, 30, 33, 43, 2, 39, 15, 44, 29, 41, 15, 24, 50]
results = [65, 7, 8, 76, 23, 56, 100, 3, 74, 48, 73, 0, 62, 37, 74, 40, 90, 42, 58, 100]

# Fit a linear regression model
model = np.polyfit(hours, results, 1)
predict = np.poly1d(model)

# Calculate R-squared
r2 = r2_score(results, predict(hours))
print("R-squared:", r2)

# Scatter plot
plt.scatter(hours, results, label='Actual data')

# Regression line plot
plt.plot(hours, predict(hours), color='red', label='Regression line')

# Labels and title
plt.xlabel('Hours')
plt.ylabel('Results')
plt.title('Scatter Plot with Regression Line')

# Show legend
plt.legend()

# Display the plot
plt.show()


# In[13]:


get_ipython().run_line_magic('pwd', '')


# In[14]:


import pandas as pd
import os

# List to store the sum of 'y' column values for each file
sum_list = []

# Directory containing CSV files
directory = './csv_files/'


# In[15]:


for filename in os.listdir(directory):
    print(f"Processing...Filename = {filename}")
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        
        # Read the CSV file using pandas
        df = pd.read_csv(file_path)
        
        # Calculate the sum of 'y' column and append to the list
        total_sum = df['y'].sum()
        sum_list.append(total_sum)
        
# Print the list of sums
print("Sum of 'y' column values for each file:", sum_list)


# In[16]:


import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.DataFrame(np.random.rand(100, 5), columns=["a", "b", "c", "d", "e"])

profile = ProfileReport(df, title="Pandas Profiling Report")

profile


# In[ ]:




