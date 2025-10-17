import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Set the axis labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


import numpy as np

datas = []
for i in [0,1,4,5,8,12,15]:
        datas.append(np.load(f'data/mixed_samples/{i}healthy.npy'))
datas = datas*40

cdict = {0: 'red', 1: 'blue', 2: 'green'}



# Rotate the axes and update
for j in range( 360*4 + 1):
    angle_norm = (j + 180) % 360 - 180
    angle = j
    i=j%10
    data = datas[j//10]

    for group in cdict.keys(): 
        filtereddata = np.where(data[i, :, 3]==group)
        ax.scatter(xs=data[i,:,0][filtereddata], 
                ys=data[i,:,1][filtereddata], 
                zs=data[i,:,2][filtereddata],
                c=random.choice(['red', 'green', 'blue', 'yellow']),
                s=0.5)
        
    # Cycle through a full rotation of elevation, then azimuth, roll, and all
    elev = azim = roll = 0
    if angle <= 360:
        elev = angle_norm
    elif angle <= 360*2:
        azim = angle_norm
    elif angle <= 360*3:
        roll = angle_norm
    else:
        elev = azim = roll = angle_norm

    # Update the axis view and title
    ax.view_init(elev, azim, roll)
    plt.title(f"Frame {i+1} of 10, heart {j//10}")

    plt.draw()
    plt.pause(.000001)

    ax.cla()