import json
import math
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def get_throughput_and_itl(model:str, num_gpus: int, bs:int, mode: str="tp"):
    tp = 1
    pp = 1
    if mode == "tp":
        tp = num_gpus
    elif mode == "pp":
        pp = num_gpus
    result_filename = f"experiments/results/result_{model}_tp{tp}_pp{pp}_r{bs}_t1000.json"
    if not os.path.exists(result_filename):
        return 0, 0
    with open(result_filename, "r") as f:
        metrics = json.load(f)
    
    throughput = metrics["total_token_throughput"] / num_gpus
    itl = np.median([itl for itls in metrics["itls"] for itl in itls]) * 1000
    # if num_gpus == 8 and mode == "pp" and bs == 1024:
    #     print(metrics["itls"])
    print(f"{mode}: {num_gpus} GPUs, bs={bs}, throughput={throughput:.2f}, itl={itl:.2f}")
    return throughput, itl

def main():
    model = sys.argv[1]
    print(f"Plotting for {model} model")

    throughput_dict = {}
    itl_dict = {}
    bs_list = [8, 16, 32, 64, 128, 256, 512, 1024]

    for num_gpus in [1, 2, 4, 8]:
        for mode in ["single", "tp", "pp"]:
            if mode == "single" and num_gpus > 1:
                continue
            if mode != "single" and num_gpus == 1:
                continue
            if mode == "single":
                key = "single"
            else:
                key = f"{mode}{num_gpus}"
            throughput_dict[key] = []
            itl_dict[key] = []
            for bs in bs_list:
                throughput, itl = get_throughput_and_itl(model, num_gpus, bs, mode)
                throughput_dict[key].append(throughput)
                itl_dict[key].append(itl)
    
    def plot_metric(metric: str):
        fig, ax = plt.subplots()
        for key in sorted(throughput_dict):
            if not throughput_dict[key]:
                continue
            if metric == "itl":
                ax.plot(bs_list, itl_dict[key], label=key, marker='o', ms=5)
            else:
                ax.plot(bs_list, throughput_dict[key], label=key, marker='o', ms=5)
        if metric == "itl":
            ax.set_title(f"ITL vs Batch Size for {model} Model")
            ax.set_ylabel("ITL (ms)")
        else:
            ax.set_title(f"Throughput vs Batch Size for {model} Model")
            ax.set_ylabel("Throughput (samples/sec/gpu)")
        ax.set_xlabel("Batch Size (log2)")
        
        ax.set_xscale("log", base=2)
        ax.set_xticks(bs_list)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # print([math.log(bs, 2) for bs in bs_list])
        ax.legend()
        plt.grid()
        fig.savefig(f"experiments/figures/{metric}_{model}.png")
        print(f"Saved experiments/figures/{metric}_{model}.png")
    
    plot_metric("throughput")
    plot_metric("itl")

if __name__ == "__main__":
    main()