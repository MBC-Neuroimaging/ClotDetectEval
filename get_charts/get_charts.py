#! /usr/bin/env python3

import os
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

'''
Makes the chart for clot detection paper
Make sure the csv is in the same location as the code :)
Install all the libraries listed above.
'''
df = pd.read_csv('Annotation_for_clot_detection - numerical info - don\'t touch.csv', header=[4])

# https://stackoverflow.com/questions/63687789/how-do-i-create-a-pie-chart-using-categorical-data-in-matplotlib
# https://stackoverflow.com/questions/25447700/annotate-bars-with-values-on-pandas-bar-plots
# https://stackoverflow.com/questions/62201784/modifying-axis-labels-for-pandas-bar-plot

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 13))

ax = axs.ravel()

# print the first 5 rows out
print(df.head())
print('All the column names: {}'.format(df.keys()))

# getting the three columns and making sure they are all numbers
df['clot_location'] = df['clot_location'].apply(pd.to_numeric, errors='coerce')
df['collateral_grade'] = df['collateral_grade'].apply(pd.to_numeric, errors='coerce')
df['hemisphere'] = df['hemisphere'].apply(pd.to_numeric, errors='coerce')
df['num_clots'] = df['num_clots'].apply(pd.to_numeric, errors='coerce')


# the following is all creating pie charts but you can see the bits and pieces that i've used
# to summarise the columns
# e.g. df.fillna(-1).groupby('collateral_grade').size() - print this statement out to see what it does
# the "df.fillna()" command simple takes any "na" numbers and turns it into -1.
# I did this purely for the pie charts to not have any rogue values putting off the colours.
# you can experiment without it and see what happens.
# the previous line (pd.tonumeric) took every entry that was no able to be turn into a number and turned it into a nan.
# nan = not a number

df.fillna(-1).groupby('collateral_grade').size().plot(
    kind='pie',
    ax=ax[3],
    labels=df.fillna(-1).groupby('collateral_grade').size(),
    wedgeprops={'linewidth': 1, 'linestyle': 'solid', 'edgecolor': 'k'},
    textprops={'fontsize': 16},
    colormap='BuPu',
    labeldistance=1.045
)
ax[3].set_ylabel('Collateral Grade', size=16)
ax[3].legend(loc=3, labels=['N/A', 'Good', 'Moderate', 'Poor'], fontsize=16)
ax[3].set_title('d)', size=18, loc='left')

df.groupby('clot_location').size().plot(
    kind='barh',
    ax=ax[1],
    color='violet',
    edgecolor='black',
    width=1/2
)
for p in ax[1].patches:
    ax[1].annotate(str(int(p.get_width())), (p.get_width() + 0.25,  p.get_y() + 0.4), size=16)


vals = ax[1].get_xticks()
y_labels = ['None', 'ICA', 'M1-MCA', 'M2-MCA', 'M3-MCA', 'ACA', 'PCA', 'Basilar']
ax[1].set_yticklabels(y_labels, size=16)
x_ticks = ax[1].get_xticks().tolist()
ax[1].set_xticklabels([str(int(p)) for p in x_ticks], size=16)
ax[1].set_ylabel('Clot Location', size=16)
ax[1].set_xlabel('')
ax[1].set_title('b)', size=18, loc='left')
ax[1].set_xlim([0, 90])
ax[1].invert_yaxis()

df.fillna(-1).groupby('hemisphere').size().plot(
    kind='pie',
    ax=ax[2],
    colors=['lightskyblue', 'limegreen', 'cadetblue', 'slateblue'],
    wedgeprops={'linewidth': 1, 'linestyle': 'solid', 'edgecolor': 'k'},
    labels=df.fillna(-1).groupby('hemisphere').size(),
    textprops={'fontsize': 16},
    labeldistance=1.06
)

ax[2].set_ylabel('Clot hemisphere', size=16)
ax[2].legend(loc=4, labels=['N/A', 'Right', 'Left', 'Basilar'], fontsize=16)
ax[2].set_title('c)', size=18, loc='left')

df.fillna(-1).groupby('num_clots').size().plot(
    kind='pie',
    ax=ax[0],
    colors=['slateblue', 'lightskyblue', 'limegreen'],
    wedgeprops={'linewidth': 1, 'linestyle': 'solid', 'edgecolor': 'k'},
    labels=df.fillna(-1).groupby('num_clots').size(),
    textprops={'fontsize': 16},
    labeldistance=1.02
)

ax[0].set_ylabel('Number of clots', size=16)
ax[0].legend(loc=4, labels=['None', 'One', 'Two'], fontsize=16)
ax[0].set_title('a)', size=18, loc='left')

# saving charts - you can replace with your file paths
# the first bit here just removes any file with the same name that exists
# the reason why I do that is that if the file is open python will NOT overwrite it and I'll get an error message
# so i found it easier to just remove it before hand instead of manually closing it.

if os.path.exists('C:/Users/fwerdiger/Documents/MBC/clot_detection/paper/pie_chart_clot_location.png'):
    os.remove('C:/Users/fwerdiger/Documents/MBC/clot_detection/paper/pie_chart_clot_location.png')
fig.savefig('C:/Users/fwerdiger/Documents/MBC/clot_detection/paper/pie_chart_clot_location.png', bbox_inches='tight')
plt.close()
