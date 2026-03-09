import matplotlib
import matplotlib.pyplot as plt
students=['mani','suraj','sai','rishi','vishnu']
marks=[45,50,55,60,70]
color=['red','black','blue','green','orange']
plt.bar(students,marks,color=color)
plt.title('Bar plot example')
plt.xlabel('students')
plt.ylabel('marks')
plt.show()
