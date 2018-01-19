import matplotlib.pyplot as plt

# Get current size
fig_size = plt.rcParams["figure.figsize"]

# Prints: [8.0, 6.0]
print( "Current size:", fig_size)

# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size


fig_size = plt.rcParams["figure.figsize"]

# Prints: [8.0, 6.0]
print( "Current size:", fig_size)
