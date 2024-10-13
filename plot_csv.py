import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_event(ax, df):

    prefills = [num_seq if df.is_prefill[i] and not pd.isna(
        num_seq) else 0 for i, num_seq in enumerate(df.num_seqs)]
    recomputes = [num_seq if df.is_prefill[i] is False and not pd.isna(
        num_seq) else 0 for i, num_seq in enumerate(df.num_seqs)]
    evictions = [1 if not pd.isna(seq_id) else 0 for seq_id in df.seq_id]
    ax.bar(df.ts, prefills, width=4, label="Prefills")
    ax.bar(df.ts, recomputes, width=4, label="Recomputes")
    ax.bar(df.ts, evictions, width=4, label="Evictions")
    
    ax.set_ylim(0, 30)

    ax.legend(loc="upper left")

def plot_batch(ax, df):

    ax.plot(df.ts, df.batch_size.ffill(), color="green", label="Batch size")

    ax.legend(loc="upper left")

def plot_result(ax, df):

    ax_r = ax.twinx()

    ln1 = ax_r.plot(df.ts, df.kv.ffill(), label="KV Cache", color="black")
    ln2 = ax_r.plot(df.ts, df.compute.ffill(),
                    label="GPU utilization", color="red")
    ln3 = ax.plot(df.ts, df.throughput.ffill(), marker='o',
                  markersize=1, linestyle="None", label="Throughput")

    lns = ln1 + ln2 + ln3
    labels = [ln.get_label() for ln in lns]
    ax.legend(lns, labels, loc="upper left")


def plot_latency(ax, df):

    ax_r = ax.twinx()

    ttfts = [ttft if ttft != 0 else np.nan for ttft in df.ttft]
    tpots = [tpot if tpot != 0 else np.nan for tpot in df.tpot]
    ln1 = ax.plot(df.ts, ttfts, marker='o', markersize=1,
                  linestyle="None", label="TTFT")
    ln2 = ax_r.plot(df.ts, tpots, marker='o', markersize=1,
                    linestyle="None", label="TPOT", color="orange")

    lns = ln1 + ln2
    labels = [ln.get_label() for ln in lns]
    ax.legend(lns, labels, loc="upper left")


def main(filename):
    fig, axs = plt.subplots(4, figsize=(8, 16))

    fields = ["ts", "kv", "compute", "throughput", "ttft",
              "tpot", "batch_size", "is_prefill", "num_seqs", "seq_id"]
    df = pd.read_csv(f"{filename}.csv", usecols=fields)
    plot_result(axs[0], df)
    plot_batch(axs[1], df)
    plot_latency(axs[2], df)
    plot_event(axs[3], df)

    for ax in axs:
        ax.set_xticks([])

    fig.savefig(f"{filename}.png", bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1])
