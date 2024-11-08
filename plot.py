import numpy as np
import matplotlib.pyplot as plt
import os

# parameters
limit = 10000

# load data
# data stored as results/ModelName-limit.npz
results = os.listdir("results")
results = [result for result in results if f"{limit}.npz" in result]

# get plt colors as array
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# plot
fig, ax = plt.subplots()
for i, result in enumerate(results):
    data = np.load(f"results/{result}")
    lossess = data["losses"]
    accuracies = data["accuracies"]
    model_name = result.split("-")[0]
    ax.plot(accuracies, label=f"{model_name} Accuracy", color=colors[i])
    ax.plot(lossess, label=f"{model_name} Loss", color=colors[i], linestyle="dashed")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss / Accuracy")
# legend smaller
plt.legend(loc="center right", fontsize="small")
plt.xticks(range(0, 40, 2))
plt.show()
