import matplotlib.pyplot as plt
from collections import OrderedDict
import json

N_RUNS = 10_000

data = OrderedDict()

with open('Taxi\Taxi_Run_Data.json', 'r') as f:
    load = json.load(f)
    for x in load:
        if(x!='Final'):
            data[int(x)] = load[x]
        else:
            data[N_RUNS] = load[x]

plt.plot(*zip(*sorted(data.items())))
plt.show()