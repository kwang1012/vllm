from datetime import datetime
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

def plot_event(ax, df):

    prefills = [num_seq if df.is_prefill[i] and not pd.isna(num_seq) else -1 for i, num_seq in enumerate(df.num_seqs)]
    recomputes = [num_seq if df.is_prefill[i] is False and not pd.isna(num_seq) else -1 for i, num_seq in enumerate(df.num_seqs)]
    evictions = [1 if not pd.isna(seq_id) else 0 for seq_id in df.seq_id]
    ts = [(datetime.strptime(t, "%H:%M:%S.%f")-datetime(1970,1,1)).total_seconds() for t in df.ts]
    ax.bar(ts, prefills, width=4, label="Prefills")
    ax.bar(ts, recomputes, width=4, label="Recomputes")
    ax.bar(ts, evictions, width=4, label="Evictions")
    ax.set_ylim(0, 30)

    ax.legend(loc="upper left")

def plot_result(ax, df):

    ax_r = ax.twinx()

    ts = [(datetime.strptime(t, "%H:%M:%S.%f")-datetime(1970,1,1)).total_seconds() for t in df.ts]
    ln1 = ax_r.plot(ts, df.kv.ffill(), label="KV Cache", color="black")
    ln2 = ax_r.plot(ts, df.compute.ffill(), label="GPU utilization", color="red")
    ln3 = ax.plot(ts, df.throughput.ffill(), marker='o', markersize=1, linestyle="None", label="Throughput")

    lns = ln1 + ln2 + ln3
    labels = [ln.get_label() for ln in lns]
    ax.legend(lns, labels, loc="upper left")

def plot_latency(ax, df):

    ax_r = ax.twinx()

    ts = [(datetime.strptime(t, "%H:%M:%S.%f")-datetime(1970,1,1)).total_seconds() for t in df.ts]
    ln1 = ax.plot(ts, df.ttft.ffill(), marker='o', markersize=1, linestyle="None", label="TTFT")
    ln2 = ax_r.plot(ts, df.tpot.ffill(), marker='o', markersize=1, linestyle="None", label="TPOT", color="orange")

    lns = ln1 + ln2
    labels = [ln.get_label() for ln in lns]
    ax.legend(lns, labels, loc="upper left")

def plot_swap(ax, df):

    ts = [(datetime.strptime(t, "%H:%M:%S.%f")-datetime(1970,1,1)).total_seconds() for t in df.ts]
    in_latency = [latency if df.is_in[i] else 0 for i, latency in enumerate(df.latency)]
    out_latency = [latency if not df.is_in[i] else 0 for i, latency in enumerate(df.latency)]
    ax.plot(ts, in_latency, label="Swap in")
    ax.plot(ts, out_latency, label="Swap out", alpha=0.7)

    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)

def main(filename):
    has_swap_file = os.path.exists(f"{filename}-swap.csv")
    fig, axs = plt.subplots(4, figsize=(8, 16 if has_swap_file else 12))
    
    fields = ["ts", "kv", "compute", "throughput", "ttft", "tpot", "is_prefill", "num_seqs", "seq_id"]
    df = pd.read_csv(f"{filename}.csv", usecols=fields)

    plot_result(axs[0], df)
    plot_latency(axs[1], df)
    plot_event(axs[2], df)

    if has_swap_file:
        fields = ["ts", "is_in", "latency"]
        df_swap = pd.read_csv(f"{filename}-swap.csv", usecols=fields)
        plot_swap(axs[3], df_swap)

    start = (datetime.strptime(df.ts[0], "%H:%M:%S.%f")-datetime(1970,1,1)).total_seconds()
    end = (datetime.strptime(df.ts[len(df.ts)-1], "%H:%M:%S.%f")-datetime(1970,1,1)).total_seconds()
    xmin = start-0.05*(end-start)
    xmax = end+0.05*(end-start)
    for ax in axs:
        ax.set_xticks([])
        ax.set_xlim(xmin, xmax)

    fig.savefig(f"{filename}.png", bbox_inches="tight")

if __name__ == "__main__":
    main(sys.argv[1])