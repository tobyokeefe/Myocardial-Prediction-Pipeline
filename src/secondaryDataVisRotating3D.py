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

i=4
for group in cdict.keys(): 
    filtereddata = np.where(data[i, :, 3]==group)
    ax.scatter(xs=data[i,:,0][filtereddata], 
               ys=data[i,:,1][filtereddata], 
               zs=data[i,:,2][filtereddata],
               c=cdict[group],
               s=2)


# Rotate the axes and update
for angle in range(0, 360*4 + 1):
    # Normalize the angle to the range [-180, 180] for display
    angle_norm = (angle + 180) % 360 - 180

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
    plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))

    plt.draw()
    plt.pause(.000001)