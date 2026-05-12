import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y =[10,14,12,18,20]
plt.plot(x, y, marker='o')
plt.title("simple line chart")
plt.xlabel("Time")
plt.ylabel("Distance")
plt.grid(True)
plt.show()
