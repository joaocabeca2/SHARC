from scipy.stats.sampling import DiscreteAliasUrn
import numpy as np

num_nodes = 10000
pv = [0.2466,0.2036,0.1405,0.1127,0.0919,0.0752,0.0556,0.0388,0.0241,0.0110]
urng = np.random.default_rng()
rng_floor = DiscreteAliasUrn(pv, random_state=urng)
wi_fi_floor =rng_floor.rvs(size=num_nodes)





import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
vec_bins = np.arange(0,11)
ax.hist(wi_fi_floor, density = True, bins = vec_bins, color = 'orange', edgecolor = "black")
plt.grid()
ax.stem(vec_bins[0:10]+0.5,pv, linefmt ='r-')
plt.show()
