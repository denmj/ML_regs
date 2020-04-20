import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# import ggplot2

L = [91,87, 95,123,98,110,112,85,71,80,69,109,90,84,75,105,
     100,99,94,90,79,86,90,93,95,100,98,80,104,77,108,90,103,89]
L.sort()
mu, std = stats.norm.fit(L)
print(mu, std)
num_bins = 5

# Freq.Dist chart
relative_freq = stats.relfreq(L, numbins=5)
relative_f_perc = relative_freq.frequency

fig, (ax1, ax2, ax3, ax3_1) = plt.subplots(nrows=1, ncols=4, figsize=(20,5))
ax1.hist(L, num_bins, color='green', alpha=0.1,
        edgecolor='b', label=('IQ counts'))
ax1.legend(loc=(0.65, 0.8))
ax1.set_xlabel('IQ ')
ax1.set_title("Frequency histogram")


ax2.hist(L, num_bins, color='blue', alpha=0.5,
        edgecolor='b', label=('IQ %'),cumulative ='True' )
ax2.legend(loc=(0.65, 0.8))
ax2.set_xlabel('IQ')
ax2.set_title("Cumulative histogram")

ax3.hist(L, weights=np.zeros_like(L) + 1/len(L), color='red', alpha=0.1,
        edgecolor='b', label=('IQ %'))

ax3.legend(loc=(0.65, 0.8))
ax3.set_xlabel('IQ ')
ax3.set_title("Relative frequency")


ax3_1.hist(L, weights=np.zeros_like(L) + 1/len(L), color='red', alpha=0.5,
        edgecolor='b', label=('IQ %'), density=True)

fit = stats.norm.pdf(L, np.mean(L), np.std(L))
ax3_1.plot(L, fit, 'k', linewidth=2, label=('Density'))

ax3_1.legend(loc=(0.65, 0.8))
ax3_1.set_xlabel('IQ ')
ax3_1.set_title("Fit: mu - {}, std - {}".format(round(mu, 2), round(std,2)))

plt.show()


"""

Variance and Standard Dev (numpy)

"""

X = [13, 10, 11, 7, 9, 11, 9]
s = [7, 3, 1, 0 , 4]
np_var = np.var(X)
std = np.std(X)

z_score = stats.zscore(X)
print("Z scores: ", z_score)

rnge = np.max(X) - np.min(X)

s_np_var = np.var(s, ddof=1)
s_std = np.std(s, ddof=1)

density_func = stats.norm.pdf(X, np.mean(X), np.std(X))
print(density_func)



# print(np_var)
# print(std)
# print(s_np_var)
# print(s_std)

"""

IQR 

"""

inter_q_range = stats.iqr(X)
# print(inter_q_range)

fig, (ax4, ax5, ax6) = plt.subplots(nrows=1, ncols=3, figsize=(18,5))
ax4.boxplot(X)
ax4.legend(loc=(0.65, 0.8))
ax4.set_xlabel('Vars ')
ax4.set_title(" Basic boxplot")
# plt.show()



