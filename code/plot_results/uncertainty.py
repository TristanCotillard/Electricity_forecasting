import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

# From simulations with uncertainties compute statistics and boxplot

INPUT_FOLDER = 'C:/Users/cotil/Desktop/COURS/Mines/2A/Sophia/Projet/main/data_elecprices/output_modified/incertitude/'

rmse_train = []
rmse_test = []
std_train = []
std_test = []

for i in range(1,6):
    local_folder = INPUT_FOLDER + '/' + str(i) + '/'

    local_rmse_train = pkl.load(open(local_folder + 'rmse_train' + '.pkl', 'rb'))
    local_rmse_test = pkl.load(open(local_folder + 'rmse_test' + '.pkl', 'rb'))
    local_std_train = pkl.load(open(local_folder + 'sd_train' + '.pkl', 'rb'))
    local_std_test = pkl.load(open(local_folder + 'sd_test' + '.pkl', 'rb'))

    rmse_train.append(local_rmse_train)
    rmse_test.append(local_rmse_test)
    std_train.append(local_std_train)
    std_test.append(local_std_test)

rmse_train = [i for j in rmse_train for i in j]
rmse_test = [i for j in rmse_test for i in j]
std_train = [i for j in std_train for i in j]
std_test = [i for j in std_test for i in j]

def stats(values, str_value):
    """
    print mean and standard deviation of values
    """
    print('max_' + str_value +  ' : ', np.max(values))
    print('min_' + str_value +  ' : ', np.min(values))
    print('mean_' + str_value +  ' : ', np.mean(values))
    print('var_mean_' + str_value +  ' : ', np.std(values)/np.mean(values) * 100)
    print('st rmse_' + str_value +  ' : ', np.std(values), '\n')

stats(rmse_train, 'rmse_train')
stats(rmse_test, 'rmse_test')
stats(std_train, 'std_train')
stats(std_test, 'std_test')

# Plot boxplots
# plt.scatter(std_train, rmse_train)
plt.boxplot(rmse_train, vert=False)
# plt.boxplot(std_train)
plt.xlabel('std')
plt.ylabel('rmse')
plt.title('Train')
plt.show()

# plt.scatter(std_test, rmse_test)
plt.boxplot(rmse_test)

plt.xlabel('std')
plt.ylabel('rmse')
plt.title('Test')
plt.show()