import json
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import functools


def main(filename):
    with open(f"{filename}.log", "r", encoding="utf-8") as f:
        lines = f.readlines()

    stage_events = {}
    execution_events = {}
    start_from = 1000
    num_events = 50
    batch_sizes = []
    bs_to_exec_time = []
    exec_time_dict = {}
    prep_time_dict = {}
    batch_exec_time_dict = {}
    batch_prep_time_dict = {}
    batch_logits_time_dict = {}
    batch_sample_time_dict = {}
    for line in lines:
        line = line.strip()
        if "Execution start time" in line:

            info = line[line.find("] ") + 2:]
            info = info.split(",")
            rank = int(info[0].split(":")[1].strip())

            if rank not in stage_events:
                stage_events[rank] = []
                execution_events[rank] = 0
            start_time = float(info[1].split(":")[1].strip())
            end_time = float(info[2].split(":")[1].strip())
            try:
                send_start_time = float(info[3].split(":")[1].strip())
                send_end_time = float(info[4].split(":")[1].strip())
            except:
                send_start_time = end_time
                send_end_time = end_time
            try:
                recv_start_time = float(info[5].split(":")[1].strip())
                recv_end_time = float(info[6].split(":")[1].strip())
            except:
                recv_start_time = start_time
                recv_end_time = start_time
            mb = info[7].split(":")[1].strip()
            bs = int(info[8].split(":")[1].strip())
            prep_time = float(info[9].split(":")[1].strip())
            logits_time = float(info[10].split(":")[1].strip())
            sample_time = float(info[11].split(":")[1].strip())
            exec_time = end_time - start_time
            bs_to_exec_time.append((bs, exec_time))
            if rank not in batch_exec_time_dict:
                batch_exec_time_dict[rank] = {}
                batch_prep_time_dict[rank] = {}
                batch_sample_time_dict[rank] = {}
                batch_logits_time_dict[rank] = {}
            if bs not in exec_time_dict:
                exec_time_dict[bs] = []
                prep_time_dict[bs] = []
            if bs not in batch_exec_time_dict[rank]:
                batch_exec_time_dict[rank][bs] = []
                batch_prep_time_dict[rank][bs] = []
                batch_sample_time_dict[rank][bs] = []
                batch_logits_time_dict[rank][bs] = []
            exec_time_dict[bs].append(1000*exec_time)
            prep_time_dict[bs].append(prep_time)
            batch_exec_time_dict[rank][bs].append(1000*exec_time - prep_time)
            batch_prep_time_dict[rank][bs].append(prep_time)
            batch_sample_time_dict[rank][bs].append(sample_time)
            batch_logits_time_dict[rank][bs].append(logits_time)

            execution_events[rank] += 1
            if start_from <= execution_events[rank] < start_from + num_events:
                stage_events[rank].append(
                    (recv_start_time, recv_end_time - recv_start_time, "tab:green", ""))
                stage_events[rank].append(
                    (start_time, end_time - start_time, "tab:blue", mb))
                stage_events[rank].append(
                    (send_start_time, send_end_time - send_start_time, "tab:orange", ""))

        elif "Num scheduled tokens" in line:
            info = line[line.find("] ") + 2:]
            info = info.split(",")

            mb = int(info[0].split(":")[1].strip())
            bs = int(info[1].split(":")[1].strip())
            total_tokens = int(info[2].split(":")[1].strip())
            batch_sizes.append((mb, bs, total_tokens))

    # rank 0: _ e s _ e s
    # rank 1: _ _ _ r e s r e s
    # rank 2: _ _ _ _ _ _ r e s r e s
    # rank 3: _ _ _ _ _ _ _ _ _ r e s r e s
    for rank, events in stage_events.items():
        if rank == len(stage_events) - 1:
            break
        for i, event in enumerate(events):
            if "orange" in event[2]:
                send_start_time = event[0]
                send_end_time = event[0] + event[1]
                recv_start_time = stage_events[rank+1][i-2][0]
                recv_end_time = stage_events[rank+1][i-2][0] + stage_events[rank+1][i-2][1]
                overlap_interval = (max(send_start_time, recv_start_time),
                                    min(send_end_time, recv_end_time))
                events[i] = (overlap_interval[0], overlap_interval[1] - overlap_interval[0],
                             event[2], event[3])
                stage_events[rank+1][i-2] = (overlap_interval[0], overlap_interval[1] - overlap_interval[0], stage_events[rank][i-2][2], stage_events[rank][i-2][3])

    print("# iterations:", execution_events[0])
    def cmp(a, b):
        return len(b[1]) - len(a[1])
    for rank, _exec_time_dict in sorted(batch_exec_time_dict.items()):
        print("===== RANK", rank, "=====")
        bs, exec_time = sorted(_exec_time_dict.items(), key=functools.cmp_to_key(cmp))[0]
        avg_exec_time = sum(exec_time) / len(exec_time)
        avg_prep_time = sum(batch_prep_time_dict[rank][bs]) / len(
            batch_prep_time_dict[rank][bs])
        if batch_logits_time_dict[rank]:
            avg_sample_time = sum(batch_sample_time_dict[rank][bs]) / len(
                batch_sample_time_dict[rank][bs])
            avg_logits_time = sum(batch_logits_time_dict[rank][bs]) / len(
                batch_logits_time_dict[rank][bs])
            print(
                f"BS: {bs}, Exec: {avg_exec_time:.1f}ms(Prep:{avg_prep_time:.1f}ms,Logits:{avg_logits_time:.1f}ms,Sample:{avg_sample_time:.1f}ms)")
        else:
            print(
                f"BS: {bs}, Exec: {avg_exec_time:.1f}ms(Prep:{avg_prep_time:.1f}ms)")
    fig, ax = plt.subplots(figsize=(48, 5))
    start_timestamp = min(e[0]
                          for events in stage_events.values() for e in events)
    stage_events_labels = [[e[3] for e in events]
                           for events in stage_events.values()]
    stage_events_colors = [[e[2] for e in events]
                           for events in stage_events.values()]
    stage_events = [[(e[0] - start_timestamp, e[1])
                     for e in events] for events in stage_events.values()]

    # green_lines = [(l[0] - start_timestamp, l[1]) for l in green_lines if l[0] > start_timestamp - 0.01]
    # itl = []
    # for i, line in enumerate(green_lines):
    #     if i != 0:
    #         itl.append(line - green_lines[i-1])


    for i, events in enumerate(stage_events):
        colors = stage_events_colors[i]
        ax.broken_barh(events, ((len(stage_events) - i - 1)
                       * 5, 4), fc=colors, ec="black")

        for eid, (x1, x2) in enumerate(events):
            ax.text(x=x1 + x2/2,
                    y=(len(stage_events) - i - 1) * 5 + 2,
                    s=stage_events_labels[i][eid],
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=18)

    # l1 = ax.vlines([l[0] for l in red_lines], 0, len(stage_events) * 5, colors="red", linestyles="dashed", label="left_time")
    # for i, (l, ve) in enumerate(red_lines):
    #     ax.text(x=l,
    #             y=(len(stage_events)) * 5 + 0.5,
    #             s=str(ve),
    #             ha='center',
    #             va='center',
    #             color='red',)
    # lns = [ l6]

    labels = ["send_time", "recv_time"]
    colors = ["tab:orange", "tab:green"]
    handles = [mpatches.Patch(color=color, label=label)
               for color, label in zip(colors, labels)]

    y_ticks = [5 * i + 2 for i in range(len(stage_events))]
    y_ticklabels = [f"Stage {i}" for i in range(len(stage_events)-1, 0)]
    ax.set_yticks(y_ticks)  # Set tick positions
    ax.set_yticklabels(y_ticklabels)
    # Add the legend
    ax.legend(handles=handles, fontsize=18)
    fig.savefig(f"{filename}-gantt.png", bbox_inches="tight")

    # fig, ax = plt.subplots()
    # ax.scatter(
    #     [bs for bs, _ in bs_to_exec_time],
    #     [exec_time for _, exec_time in bs_to_exec_time],
    #     marker="o",
    #     linestyle="-",
    #     color="tab:blue",
    # )
    # ax.set_xlabel("Batch Size", fontsize=18)
    # ax.set_ylabel("Execution Time (s)", fontsize=18)
    # ax.set_title("Batch Size vs Execution Time", fontsize=18)
    # fig.savefig(f"{filename}-exec_time.png", bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1])
