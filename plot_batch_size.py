import json
import matplotlib.pyplot as plt
import sys
import numpy as np

def main(filename):
    with open(f"results/{filename}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    fig, ax = plt.subplots()

    X = np.arange(len(data))
    ax.plot(X, data, label="batch size")

    print(np.mean(data))

    ax.legend()
    fig.savefig(f"results/{filename}.png")

if __name__ == "__main__":
    main(sys.argv[1])