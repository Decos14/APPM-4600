import matplotlib.pyplot as plt
import numpy as np

fb = lambda x: (2.0*x)/3.0 + 1.0/(x**2)
y = [np.random.uniform(0.5,1.5)]
for i in range(6):
    y.append(fb(y[i]))
    
fix = 3**(1/3)
plt.plot(y)
plt.plot([fix for _ in range(len(y))])

plt.show()
lam = []
for i in range(len(y)-1):
    val = np.abs(y[i+1]-fix)/np.abs(y[i]-fix)**2
    lam.append(val)
    
plt.plot(lam)
plt.ylabel("lambda at n")
plt.xlabel("n")
plt.title("lamda = "+ str(lam[-1]))
print(lam[-1])
plt.savefig("4_b.png")