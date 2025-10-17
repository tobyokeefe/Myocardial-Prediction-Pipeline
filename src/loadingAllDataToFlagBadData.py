import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import numpy as np

from os import listdir
from os.path import isfile, join


candidate_files = [f for f in listdir('data/mixed_samples/') if isfile(join('data/mixed_samples/', f))]

accepted_files = []
rejected_files = []

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

cdict = {0: 'red', 1: 'blue', 2: 'green'}

for file in candidate_files[0:len(candidate_files)]:
    data = np.load(join('data/mixed_samples/', file))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for i in range(10):
        for group in cdict.keys(): 
            filtereddata = np.where(data[i, :, 3]==group)
            ax.scatter(xs=data[i,:,0][filtereddata], 
                    ys=data[i,:,1][filtereddata], 
                    zs=data[i,:,2][filtereddata],
                    c=cdict[group],
                    s=0.5)

        plt.draw()
        plt.pause(.000001)
        ax.cla()
    
    accept = input("Press enter to accept, press (n) ((or anything else)) to reject")
    if accept == "":
        accepted_files.append(file)
    else:
        print(rejected_files)
        print(f"^ previous rejected files, now {file} has been added to this")
        rejected_files.append(file)

print("------------Accepted files------------")
print(accepted_files)
print("------------------------------------")
print()
print("------------Rejected files------------")
print(rejected_files)
print("------------------------------------")


"""
['426pMI.npy', '445pMI.npy', '846healthy.npy', '289healthy.npy', '208healthy.npy', '595healthy.npy', '410pMI.npy', '390healthy.npy', '814pMI.npy', '693healthy.npy', '754pMI.npy', '841pMI.npy', '8healthy.npy', '174pMI.npy', '853pMI.npy', '175healthy.npy', '875pMI.npy', '308healthy.npy', '734healthy.npy', '793healthy.npy', '424pMI.npy', '573healthy.npy', '258healthy.npy', '21pMI.npy', '163healthy.npy', '172pMI.npy', '776pMI.npy', '335healthy.npy', '139pMI.npy', '794healthy.npy', '508pMI.npy', '502healthy.npy', '699pMI.npy', '292healthy.npy', '479healthy.npy', '68pMI.npy', '280pMI.npy', '774pMI.npy', '621healthy.npy', '385pMI.npy', '70healthy.npy', '485healthy.npy', '448pMI.npy', '378pMI.npy', '217pMI.npy', '591pMI.npy', '738pMI.npy', '638pMI.npy', '345healthy.npy', '673pMI.npy', '86pMI.npy', '313pMI.npy', '503healthy.npy', '899healthy.npy', '80healthy.npy', '539healthy.npy', '138pMI.npy', '91healthy.npy', '75pMI.npy', '119healthy.npy', '20pMI.npy', '872pMI.npy', '181healthy.npy', '597pMI.npy', '203pMI.npy', '658healthy.npy', '439healthy.npy', '564pMI.npy']
"""


     







