import matplotlib.pyplot as plt
Activities = ['Eating', 'Sleeping', 'Working', 'Playing']
hours = [2, 8, 6, 4]
plt.pie(hours,labels=Activities)
plt.title("Daily Activities")
plt.show()
