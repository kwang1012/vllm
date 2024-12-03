import json
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main(filename):
    with open(f"{filename}.log", "r", encoding="utf-8") as f:
        lines = f.readlines()

    target_virtual_engine = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    stage_events = None
    skip_first_50 = True
    skip_counter = 0
    eid = 0
    is_last_prefill = False
    red_lines = []
    blue_lines = []
    green_lines = []
    skip_first = True
    black_lines = []
    yellow_lines = []
    purple_lines = []
    for l_n, line in enumerate(lines):
        line = line.strip()
        if "Avg prompt throughput" in line:
            if skip_first_50:
                skip_counter += 1
                if skip_counter == 5:
                    skip_first_50 = False
                continue
            info = line[line.find("] ") + 2:]
            info = info.split(",")

            virtual_engine = int(info[0].split(":")[1].strip())
            tpot = float(info[9].split(":")[1].strip())
            stage_info = info[13].split(":")[1].strip().rstrip(".")
            stage_info = [json.loads(x.replace(";", ":").replace(
                "*", ",")) for x in stage_info.split(">")]

            if tpot == 0:
                is_last_prefill = True
                continue
            if stage_events is None:
                stage_events = [[] for _ in stage_info]

            for i, stage in enumerate(stage_info):
                start_time = stage["exec_timestamp"]
                end_time = stage["exec_time"]

                if stage["enter_timestamp"] and stage["enter_timestamp"] >= stage["prev_exec_timestamp"]:
                    stage_events[i].append(
                        (stage["enter_timestamp"], stage["enter_time"], "tab:brown", f"{virtual_engine}"))
                    if not is_last_prefill:
                        stage_events[i].append(
                            (stage["prev_exec_timestamp"], stage["enter_timestamp"] - stage["prev_exec_timestamp"], "tab:pink", f"{virtual_engine}"))
                elif not is_last_prefill:
                    stage_events[i].append(
                        (stage["prev_exec_timestamp"], stage["prev_exec_time"], "tab:pink", f"{virtual_engine}"))
                stage_events[i].append(
                    (stage["start_timestamp"], stage["prep_time"], "tab:green", f"{virtual_engine}"))
                if stage["recv_timestamp"] is not None:
                    stage_events[i].append(
                        (stage["recv_timestamp"], stage["recv_time"], "tab:red", f"{virtual_engine}"))
                stage_events[i].append(
                    (start_time, end_time, "tab:blue", f"{virtual_engine}"))
                
                # black_lines.append((stage["multiproc_end_timestamp"], virtual_engine))
            print(l_n)

            eid += 1
            is_last_prefill = False

        elif "left time" in line:
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            virtual_engine = int(info[0].split(":")[1].strip())
            # if virtual_engine != target_virtual_engine:
            #     continue
            left_time = float(info[1].split(":")[1].strip())
            red_lines.append((left_time, virtual_engine))

        elif "timestamp before execution" in line:
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            virtual_engine = int(info[0].split(":")[1].strip())
            # if virtual_engine != target_virtual_engine:
            #     continue
            enter_time = float(info[1].split(":")[1].strip())
            blue_lines.append((enter_time, virtual_engine))

        elif "process timestamp" in line:
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            virtual_engine = int(info[0].split(":")[1].strip())
            # if virtual_engine != target_virtual_engine:
            #     continue
            proc_timestamp = float(info[1].split(":")[1].strip())
            green_lines.append((proc_timestamp, virtual_engine))

        elif "executor start" in line:
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            virtual_engine = int(info[0].split(":")[1].strip())
            check_timestamp = float(info[1].split(":")[1].strip())
            interval = float(info[2].split(":")[1].strip())
            yellow_lines.append((check_timestamp, virtual_engine))
        elif "executor end" in line:
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            virtual_engine = int(info[0].split(":")[1].strip())
            end_timestamp = float(info[1].split(":")[1].strip())
            purple_lines.append((end_timestamp, virtual_engine))

        if eid > 20:
            break

    fig, ax = plt.subplots(figsize=(48, 5))
    start_timestamp = min(e[0] for events in stage_events for e in events)
    stage_events_labels = [[e[3] for e in events] for events in stage_events]
    stage_events_colors = [[e[2] for e in events] for events in stage_events]
    stage_events = [[(e[0] - start_timestamp, e[1])
                     for e in events] for events in stage_events]
    
    red_lines = [(l[0] - start_timestamp, l[1]) for l in red_lines if l[0] > start_timestamp - 0.01]
    blue_lines = [(l[0] - start_timestamp, l[1]) for l in blue_lines if l[0] > start_timestamp - 0.01]
    green_lines = [(l[0] - start_timestamp, l[1]) for l in green_lines if l[0] > start_timestamp - 0.01]
    black_lines = [(l[0] - start_timestamp, l[1]) for l in black_lines if l[0] > start_timestamp - 0.01]
    purple_lines = [(l[0] - start_timestamp, l[1]) for l in purple_lines if l[0] > start_timestamp - 0.01]
    yellow_lines = [(l[0] - start_timestamp, l[1]) for l in yellow_lines if l[0] > start_timestamp - 0.01]

    # itl = []
    # for i, line in enumerate(green_lines):
    #     if i != 0:
    #         itl.append(line - green_lines[i-1])
    for i, events in enumerate(stage_events):
        colors = stage_events_colors[i]
        ax.broken_barh(events, ((len(stage_events) - i - 1)
                       * 5, 4), facecolors=colors)

        for eid, (x1, x2) in enumerate(events):
            ax.text(x=x1 + x2/2,
                    y=(len(stage_events) - i - 1) * 5 + 2,
                    s=stage_events_labels[i][eid],
                    ha='center',
                    va='center',
                    color='white',)

    # l1 = ax.vlines([l[0] for l in red_lines], 0, len(stage_events) * 5, colors="red", linestyles="dashed", label="left_time")
    # for i, (l, ve) in enumerate(red_lines):
    #     ax.text(x=l,
    #             y=(len(stage_events)) * 5 + 0.5,
    #             s=str(ve),
    #             ha='center',
    #             va='center',
    #             color='red',)
    # l2 = ax.vlines([l[0] for l in blue_lines], 0, len(stage_events) * 5, colors="blue", linestyles="dashed", label="start_time")
    # for i, (l, ve) in enumerate(blue_lines):
    #     ax.text(x=l,
    #             y=(len(stage_events)) * 5 + 0.5,
    #             s=str(ve),
    #             ha='center',
    #             va='center',
    #             color='blue',)
    l3 = ax.vlines([l[0] for l in green_lines], 0, len(stage_events) * 5, colors="green", linestyles="dashed", label="processed_time")
    for i, (l, ve) in enumerate(green_lines):
        ax.text(x=l,
                y=(len(stage_events)) * 5 + 0.5,
                s=str(ve),
                ha='center',
                va='center',
                color='green',)
    # l4 = ax.vlines([l[0] for l in black_lines], 0, len(stage_events) * 5, colors="black", linestyles="dashed", label="multiproc_end_time")
    # for i, (l, ve) in enumerate(black_lines):
    #     ax.text(x=l,
    #             y=(len(stage_events)) * 5 + 0.5,
    #             s=str(ve),
    #             ha='center',
    #             va='center',
    #             color='black',)
    # l5 = ax.vlines([l[0] for l in purple_lines], 0, len(stage_events) * 5, colors="purple", linestyles="dashed", label="executor end")
    # for i, (l, ve) in enumerate(purple_lines):
    #     ax.text(x=l,
    #             y=(len(stage_events)) * 5 + 0.5,
    #             s=str(ve),
    #             ha='center',
    #             va='center',
    #             color='purple',)
    l6 = ax.vlines([l[0] for l in yellow_lines], 0, len(stage_events) * 5, colors="grey", linestyles="dashed", label="executor start")
    for i, (l, ve) in enumerate(yellow_lines):
        ax.text(x=l,
                y=(len(stage_events)) * 5 + 0.5,
                s=str(ve),
                ha='center',
                va='center',
                color='grey',)
    lns = [ l6]

    labels = ["exec_time", "prep_time", "recv_time", "prev_exect_time", "enter_time"] + [ln.get_label() for ln in lns]
    colors = ["tab:blue", "tab:green", "tab:red", "tab:pink", "tab:brown"]
    handles = [mpatches.Patch(color=color, label=label)
               for color, label in zip(colors, labels)] + lns

    y_ticks = [5 * i + 2 for i in range(len(stage_events))]
    y_ticklabels = [f"Stage {i}" for i in range(len(stage_events)-1, 0)]
    ax.set_yticks(y_ticks)  # Set tick positions
    ax.set_yticklabels(y_ticklabels)
    # Add the legend
    ax.legend(handles=handles)
    fig.savefig(f"{filename}-gantt.png", bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1])
