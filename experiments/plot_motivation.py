import json
import matplotlib.pyplot as plt
import numpy as np

def main():

    model = "8b"
    bs = 512
    num_gpus_list = [1, 2, 4, 8]
    # Read from results
    results = {
        "TP": {
            "throughput": [0],
            "itl": []
        },
        "PP": {
            "throughput": [0],
            "itl": []
        }
    }
    for num_gpus in num_gpus_list:
        result_filename = f"experiments/results/result_{model}_tp{num_gpus}_pp1_r{bs}_t1000.json"
        with open(result_filename, 'r') as f:
            metrics = json.load(f)
            throughput = metrics["total_token_throughput"]
            itl = np.mean([itl for itls in metrics["itls"] for itl in itls]) * 1000
            results["TP"]["throughput"].append(throughput)
            results["TP"]["itl"].append(itl)
        result_filename = f"experiments/results/result_{model}_tp1_pp{num_gpus}_r{bs}_t1000.json"
        with open(result_filename, 'r') as f:
            metrics = json.load(f)
            throughput = metrics["total_token_throughput"]
            itl = np.mean([itl for itls in metrics["itls"] for itl in itls]) * 1000
            results["PP"]["throughput"].append(throughput)
            results["PP"]["itl"].append(itl)
    
    num_gpus_list_zero = [0] + num_gpus_list
    # Plotting
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(num_gpus_list_zero, results["TP"]["throughput"], label="TP", marker='o')
    ax.plot(num_gpus_list_zero, results["PP"]["throughput"], label="PP", marker='o')
    # plot linear
    ax.plot(num_gpus_list_zero, [results["PP"]["throughput"][1]*num_gpus for num_gpus in num_gpus_list_zero], label="Linear", linestyle='--')
    ax.set_xlabel("#GPUs", fontsize=24)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=24)
    # ax.set_title(f"Throughput vs Number of GPUs for {model} model")
    ax.locator_params(axis='y', nbins=4)
    ax.set_xticks(num_gpus_list_zero)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(None, 1.1*results["TP"]["throughput"][1] * num_gpus_list_zero[-1])
    ax.legend(fontsize=24)
    fig.savefig("experiments/figures/throughput_vs_num_gpus.pdf", bbox_inches='tight')

    # Plotting ITL
    fig, ax = plt.subplots()
    ax.plot(num_gpus_list, results["TP"]["itl"], label="TP", marker='o')
    ax.plot(num_gpus_list, results["PP"]["itl"], label="PP", marker='o')
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("ITL (ms)")
    ax.set_title(f"ITL vs Number of GPUs for {model} model")
    ax.legend()
    fig.savefig("experiments/figures/itl_vs_num_gpus.pdf", bbox_inches='tight')
    
if __name__ == "__main__":
    main()
# This script generates a simple plot to visualize the sine and cosine functions.