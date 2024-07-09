import json
import matplotlib.pyplot as plt
import sys
import numpy as np

def main(filename):
    with open(f"results/{filename}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    in_spds = [d[0] for d in data]
    out_spds = [d[1] for d in data]
    fig, ax = plt.subplots()

    X = np.arange(len(data))
    ax.plot(X, in_spds, label="input")
    ax.plot(X, out_spds, label="output")

    ax.legend()
    fig.savefig(f"results/{filename}.png")

if __name__ == "__main__":
    main(sys.argv[1])