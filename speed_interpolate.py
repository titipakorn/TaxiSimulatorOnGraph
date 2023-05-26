#%%
import numpy as np
import pandas as pd

# Create an example dataset with speed data at 5-minute intervals
time = pd.date_range('2022-01-01 00:00:00', '2022-01-01 01:00:00', freq='5min')
speed = np.array([60, 65, 70, 68, 72, 75, 78, 80, 85, 90, 80])

# Create a new time range for 1-minute intervals
new_time = pd.date_range('2022-01-01 00:00:00', '2022-01-01 01:00:00', freq='1min')

# Create a new array to store the interpolated speed data
new_speed = np.zeros(len(new_time))

#%%
# Interpolate the speed data using linear interpolation
for i in range(len(speed)-1):
    # Determine the number of intervals to add between each original data point
    num_intervals = int((time[i+1]-time[i]).total_seconds() / 60) - 1
    print('num_intervals',num_intervals)
    # Calculate the speed for each new interval using linear interpolation
    for j in range(num_intervals):
        fraction = (j+1) / (num_intervals+1)
        print(j,fraction)
        new_speed[i*num_intervals+j+i+1] = (1-fraction) * speed[i] + fraction * speed[i+1]
        print(i*num_intervals+j+i+1,)

# Add the last value of the original speed data to the new speed array
new_speed[-1] = speed[-1]

# Create a new DataFrame to store the interpolated data
df = pd.DataFrame({'time': new_time, 'speed': new_speed})

# Print the interpolated speed data
print(df)
# %%
