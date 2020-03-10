import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat


sent = [5, 7, 13, 9, 1, 13]
rec = [10, 12, 14, 18, 6, 4]

fig = plt.figure(figsize=(10, 4))

plt.scatter(sent, rec)
plt.show()


corr_cef = np.corrcoef(sent, rec)
print(corr_cef)