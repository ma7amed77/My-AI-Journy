import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("P1_LinearRegression/LR_Scores.csv")
def Calc_Total_Error(w, b, points):
    m = len(points)
    total = 0
    for i in range(m):
        error = (points.Y[i]-(w*points.X[i]-b)) ** 2.0
        total += error
    total/=(2*m)
    return total

def Gradient_Descent(w_old, b_old, points, l):
    m = len(points)
    grad_b = 0.0
    grad_w = 0.0
    for i in range(m):
        x = points.X[i]
        y = points.Y[i]
        grad_b += -(y - (w_old * x - b_old)) / m
        grad_w += -x * (y - (w_old * x - b_old)) / m 
    w = w_old - grad_w * l
    b = b_old - grad_b * l
    return w,b

w = 0.0
b = 0.0
for i  in range(1000):
    w,b=Gradient_Descent(w,b,data,.0001)
    if i%1000==0:
        print(f"Loading {i}")

plt.scatter(data.X,data.Y,color="black")
plt.plot(list(range(0,11)),[w*x+b for x in range(0,11)])
plt.show()
