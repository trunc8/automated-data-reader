import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('graph_coordinates.csv','r') as csvfile:
  plots = csv.reader(csvfile, delimiter=',')
  for row in plots:
    x.append(float(row[0]))
    y.append(float(row[1]))

plt.plot(x,y, label='Extracted graph')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graph Reader')
plt.legend()
plt.show()