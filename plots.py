import os
import matplotlib.pyplot as plt
import numpy as np

# initialize the results of the experiements
# arrhythmia
prc_gr_arr = [0.5606, 0.5976, 0.5986, 0.6053, 0.6109, 0.6219, 0.6076, 0.6115]
prc_ac_arr = [0.5606, 0.5976, 0.5719, 0.5961, 0.6041, 0.5792, 0.6019, 0.6115]
prc_rd_arr = [0.5606, 0.5993, 0.5788, 0.6356, 0.5908, 0.6094, 0.6202, 0.6115]

# letter
prc_gr_lt = [0.6003, 0.7300, 0.7234, 0.7199, 0.7169, 0.7285, 0.7323, 0.7376]
prc_ac_lt = [0.6003, 0.7300, 0.7210, 0.7272, 0.7477, 0.7302, 0.7308, 0.7376]
prc_rd_lt = [0.6003, 0.6653, 0.7140, 0.7248, 0.7397, 0.7302, 0.7232, 0.7376]

# cardio
prc_gr_car = [0.9304, 0.9290, 0.9374, 0.9385, 0.9296, 0.9351, 0.9327, 0.9332]
prc_ac_car = [0.9304, 0.9290, 0.9314, 0.9315, 0.9337, 0.9354, 0.9331, 0.9332]
prc_rd_car = [0.9304, 0.9297, 0.9342, 0.9364, 0.9315, 0.9267, 0.9248, 0.9332]

# speech
prc_gr_sp = [0.1455, 0.2658, 0.2733, 0.3203, 0.3290, 0.3107, 0.3355, 0.2492]
prc_ac_sp = [0.1455, 0.2658, 0.2367, 0.2630, 0.3103, 0.2983, 0.3255, 0.2492]
prc_rd_sp = [0.1455, 0.1356, 0.1814, 0.2101, 0.3194, 0.3053, 0.2940, 0.2492]

# mammography
prc_gr_ma = [0.6974, 0.6853, 0.6719, 0.6720, 0.6620, 0.6717, 0.6687, 0.6673]
prc_ac_ma = [0.6974, 0.6853, 0.6915, 0.6841, 0.6965, 0.6631, 0.6655, 0.6673]
prc_rd_ma = [0.6974, 0.6812, 0.6823, 0.6649, 0.6693, 0.6619, 0.6654, 0.6673]

# x-axis
x = [0, 1, 5, 10, 30, 50, 70, 100]

# main plots
fig = plt.figure(figsize=(8, 10))
lw = 2

ax = fig.add_subplot(511)

plt.plot(x, prc_rd_arr, color='black', linestyle='-.', marker='s',
         lw=lw, label='Random Selection')
plt.plot(x, prc_gr_arr, color='blue', linestyle='--', marker='^',
         lw=lw, label='Balance Selection')
plt.plot(x, prc_ac_arr, color='red', linestyle='-', marker='o',
         lw=lw, label='Accurate Selection')

plt.xlim([-0.5, 100.5])
plt.xticks(np.arange(0, 100, 5))
plt.ylabel('Precision@n', fontsize=12)
plt.title('Arrhythmia', fontsize=12)
plt.legend(loc="lower right")

#########################################################################
ax = fig.add_subplot(512)
plt.plot(x, prc_rd_lt, color='black', linestyle='-.', marker='s',
         lw=lw, label='Random Selection')
plt.plot(x, prc_gr_lt, color='blue', linestyle='--', marker='^',
         lw=lw, label='Balance Selection')
plt.plot(x, prc_ac_lt, color='red', linestyle='--', marker='o',
         lw=lw, label='Accurate Selection')

plt.xlim([-0.5, 100.5])
plt.xticks(np.arange(0, 100, 5))
plt.ylabel('Precision@n', fontsize=12)
plt.title('Letter', fontsize=12)
plt.legend(loc="lower right")

#########################################################################
ax = fig.add_subplot(513)
plt.plot(x, prc_rd_car, color='black', linestyle='-.', marker='s',
         lw=lw, label='Random Selection')
plt.plot(x, prc_gr_car, color='blue', linestyle='--', marker='^',
         lw=lw, label='Balance Selection')
plt.plot(x, prc_ac_car, color='red', linestyle='--', marker='o',
         lw=lw, label='Accurate Selection')

plt.xlim([-0.5, 100.5])
plt.xticks(np.arange(0, 100, 5))
plt.ylabel('Precision@n', fontsize=12)
plt.title('Cardio', fontsize=12)
plt.legend(loc="lower right")

#########################################################################
ax = fig.add_subplot(514)
plt.plot(x, prc_rd_sp, color='black', linestyle='-.', marker='s',
         lw=lw, label='Random Selection')
plt.plot(x, prc_gr_sp, color='blue', linestyle='--', marker='^',
         lw=lw, label='Balance Selection')
plt.plot(x, prc_ac_sp, color='red', linestyle='--', marker='o',
         lw=lw, label='Accurate Selection')

plt.xlim([-0.5, 100.5])
plt.xticks(np.arange(0, 100, 5))
plt.ylabel('Precision@n', fontsize=12)
plt.title('Speech', fontsize=12)
plt.legend(loc="lower right")
#########################################################################
ax = fig.add_subplot(515)
plt.plot(x, prc_rd_ma, color='black', linestyle='-.', marker='s',
         lw=lw, label='Random Selection')
plt.plot(x, prc_gr_ma, color='blue', linestyle='--', marker='^',
         lw=lw, label='Balance Selection')
plt.plot(x, prc_ac_ma, color='red', linestyle='--', marker='o',
         lw=lw, label='Accurate Selection')

plt.xlim([-0.5, 100.5])
plt.xticks(np.arange(0, 100, 5))
plt.xlabel('Number of Selected ODS')
plt.ylabel('Precision@n', fontsize=12)
plt.title('Mammography', fontsize=12)
plt.legend(loc="upper right")

#########################################################################
plt.tight_layout()
plt.savefig(os.path.join('figs', 'results.png'), dpi=300)
plt.show()
