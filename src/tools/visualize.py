import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
plt.style.use('ggplot')

# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
# ax1.plot(x, y)
# ax1.set_title('Sharing x per column, y per row')
# ax2.scatter(x, y)
# ax3.scatter(x, 2 * y ** 2 - 1, color='r')
# ax4.plot(x, 2 * y ** 2 - 1, color='r')

def plot(data,name):
    data = np.matrix.transpose(data)
    var = [np.var(data[i,:]) for i in range(len(data))]
    mean = [np.mean(data[i,:]) for i in range(len(data))]
    print np.min(mean)
    ind = [i for i in range(len(data))]
    #return ind, mean
    line, = plt.plot(ind,mean,label=name)
    return line
    # plt.errorbar(ind,mean,var,errorevery=20,   capsize=2,
    #     elinewidth=1,
    #     markeredgewidth=1)


fixcupandgoal = np.genfromtxt("qtable_True_True.csv",dtype=float, delimiter=',')
notfixcupandnotgoal = np.genfromtxt("qtable_False_False.csv",dtype=float, delimiter=',')
fixcupandnotgoal = np.genfromtxt("qtable_True_False.csv",dtype=float, delimiter=',')
notfixcupandgoal = np.genfromtxt("qtable_False_True.csv",dtype=float, delimiter=',')

data_array = {"fixedcupandgoal":fixcupandgoal,"notfixedcupandgoal":notfixcupandgoal,"fixedcupandnotfixedgoal":fixcupandnotgoal,"notfixedcupandnotfixedgoal":notfixcupandnotgoal}
fig = plt.subplot(111)
# fig.suptitle('Learning ')
plt.xlabel('episodes')
plt.ylabel('reward')
fig.set_ylim([-0.3,1.2])
order = ["fixedcupandgoal","notfixedcupandgoal"	,'fixedcupandnotfixedgoal',"notfixedcupandnotfixedgoal"]
for x in order:
    line = plot(data_array[x],x)
    
    plt.legend(handler_map={line: HandlerLine2D(numpoints=4)})
    plt.pause(5.0)
