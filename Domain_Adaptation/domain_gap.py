import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d_list = np.load('./dis1.npz')['d']
for ii in range(3):
    print(d_list[ii].max(), d_list[ii].mean(), d_list[ii].min())
# d_list = 1 / (1 + np.exp(-d_list))
# print(d_list.max(), d_list.mean(), d_list.min())
print(d_list.shape)
d_list = np.mean(d_list, (2, 3))
d_list = np.reshape(d_list, (3, -1))
print(d_list.shape)
d_list_pd = pd.DataFrame({'src_rec': d_list[0], 'src_real': d_list[1], 'tg_real': d_list[2]})
# d_list_pd = pd.DataFrame(data=d_list)

name_list = ['source DA',
             'source',
             'target']

d_list_pd.src_rec.plot(kind='kde', color='mediumblue', label=name_list[0])  #
d_list_pd.src_real.plot(kind='kde', color='green', label=name_list[1])  #
d_list_pd.tg_real.plot(kind='kde', color='red', label=name_list[2])  #

plt.legend()

# d_list_pd.src_rec.plot(kind='hist', bins=20, color='skyblue', edgecolor='black', density=True)  # , stacked=True
# d_list_pd.src_real.plot(kind='hist', bins=20, color='palegreen', edgecolor='black', density=True)  # , stacked=True
# d_list_pd.tg_real.plot(kind='hist', bins=20, color='lightcoral', edgecolor='black', density=True)  # , stacked=True

# plt.xlabel('年龄')
# plt.ylabel('核密度值')
# plt.title('乘客年龄分布')
plt.xlim([-0.1, 1.1])
plt.show()
