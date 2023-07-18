import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

params = {
    "figure.figsize": (5, 5),
    "legend.fontsize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "font.family": "Liberation Mono"
}

plt.rcParams.update(params)
plt.style.use("seaborn-whitegrid")
sns.set_style("white")

fig, ax1 = plt.subplots()

lambda_2 = [0.5, 1, 2, 4, 8]
asr = [7.48, 7.2, 4.27, 2.89, 1.83]
ca  = [17.8, 18.14, 17.8, 18, 17.4]

plt.plot(lambda_2, ca, marker = 'o')
# ax2 = ax1.twinx()
# ax1.plot(lambda_2, asr, 'g-')
# ax2.plot(lambda_2, ca, 'b-')

plt.title('CA of SelFi Defense (Blended Attack)')
plt.xlabel(r'$\lambda_2$')
plt.ylabel('Accuracy (%)')
plt.grid()
plt.tight_layout()
plt.savefig('ca_blended.png')