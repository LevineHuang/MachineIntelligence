import pandas
import matplotlib.pyplot as plt
# 载入数据，去除脚注信息
dataset = pandas.read_csv('../data/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
plt.plot(dataset)
plt.show()