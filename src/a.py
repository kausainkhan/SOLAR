from scipy import stats # to do linear regression
import csv # to read the CSV files
import matplotlib.pyplot as plt # to plot the data
import numpy as np # dependency for scipy
  
# declaring two empty arrays to store the CSV data and to do the calculations
x = []
y = []

# reading the CSV file
with open('weather_data.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        x.append(row[0])
        # 8 = Column Number in the CSV file (Cloud Cover in this case)
        y.append(row[8])

# removing the header for the CSV (the array before removing the first index element
# should look like ['header-name', value1, value2, value3]) so that it can convert 
# the values into integers
x.pop(0)
x = (np.array(x)).astype(np.float)
y.pop(0)
y = (np.array(y)).astype(np.float)

# co-ordinate geometry stuff
slope, intercept, r, p, std_err = stats.linregress(x, y)

def func(x):
  return slope * x + intercept

# the time given in the CSV file is in minutes, and they have used integers and not a 
# format like "00:00", so the prediction should be in the same format, explaining the 410
prediction = func(410)
print(prediction)

# plotting the CSV file
plt.scatter(x, y, color = 'r')
plt.show()