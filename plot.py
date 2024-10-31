import numpy as np
import matplotlib.pyplot as plt
import os

# parameters
limit = 10000

# load data
# data stored as results/ModelName-limit.npz
results = os.listdir("results")
results = [result for result in results if f"{limit}.npz" in result]

# plot
fig, ax = plt.subplots()
for result in results:
    data = np.load(f"results/{result}")
    lossess = data["losses"]
    accuracies = data["accuracies"]
    model_name = result.split("-")[0]
    ax.plot(lossess, label=f"{model_name} Loss")
    ax.plot(accuracies, label=f"{model_name} Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss / Accuracy")
# legend smaller
plt.legend(loc="center right", fontsize="small")
plt.xticks(range(0, 40, 2))
plt.show()
