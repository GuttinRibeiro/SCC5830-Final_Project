import numpy as np
import matplotlib.pyplot as plt

values = []
with open('histogram.txt', 'r') as reader:
    line = reader.readline()
    while line != '':
        values.append(float(line.strip()))
        line = reader.readline()

values = np.array(values)

fig, ax = plt.subplots()
hist = ax.hist(values, bins=16, range=(0.02, 0.34))
ax.xaxis.set_major_locator(plt.IndexLocator(base=0.02, offset=0.02))
plt.xlabel('MAE')
plt.grid(True, axis='y')
plt.show()