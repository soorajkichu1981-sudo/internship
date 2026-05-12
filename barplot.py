import matplotlib.pyplot as plt
students  = ['Alice', 'Bob', 'Charlie', 'David', 'Eva']
Marks = [85,70,90,65,95]
plt.bar(students,Marks,color=['blue','orange','green','red','purple'])
plt.title("Student Marks")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.show()
