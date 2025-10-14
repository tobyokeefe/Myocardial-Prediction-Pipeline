import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Set the axis labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


import numpy as np
data = np.load('data/mixed_samples/0healthy.npy')


cdict = {0: 'red', 1: 'blue', 2: 'green'}



# Rotate the axes and update
for j in range(10*4):
    i=j%10
    for group in cdict.keys(): 
        filtereddata = np.where(data[i, :, 3]==group)
        ax.scatter(xs=data[i,:,0][filtereddata], 
                ys=data[i,:,1][filtereddata], 
                zs=data[i,:,2][filtereddata],
                c=cdict[group],
                s=0.5)
        
    # Update the axis view and title
    ax.view_init(40, 40, 0)
    plt.title(f"Frame {i+1} of 10")

    plt.draw()
    plt.pause(.000001)

    ax.cla()