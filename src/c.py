import matplotlib.pyplot as plt
import csv
  
Names = []
Values = []
  
with open('weather_data.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    print(lines)
    for row in lines:
        Names.append(row[0])
        # 11 = Column Number
        Values.append(row[11])
  
plt.scatter(Names, Values, color = 'g')
plt.show()