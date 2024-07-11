import csv
from datetime import datetime
import sys

def main(filename):
    with open(f"{filename}.log", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print_kv_cache = False
    print_model_weights = False

    result_dict = []

    throughputs = []
    latencies = []
    e2e_latency = None

    for line in lines:
        line = line.strip()
        if "Allocating" in line and not print_kv_cache:
            print(line)
            print_kv_cache = True
        elif "Loading" in line and not print_model_weights:
            print(line)
            print_model_weights = True
        elif "Avg prompt throughput" in line:
            result = {}
            time_str = line.split(" ")[2]
            result["ts"] = time_str
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            virtual_engine = int(info[0].split(":")[1].strip())
            if virtual_engine != 0:
                continue
            prompt_throughput = float(info[1].split(":")[1].strip().split(" ")[0])
            generation_throughput = float(info[2].split(":")[1].strip().split(" ")[0])
            ttft = float(info[8].split(":")[1].strip())
            tpot = float(info[9].split(":")[1].strip())
            kv_cache = float(info[6].split(":")[1].strip().rstrip("%"))
            gpu_utilization = int(info[10].split(":")[1].strip().rstrip("."))
            result["throughput"] = generation_throughput if prompt_throughput == 0 else prompt_throughput
            result["kv"] = kv_cache
            result["compute"] = gpu_utilization
            result["ttft"] = ttft
            result["tpot"] = tpot
            result_dict.append(result)
            if generation_throughput != 0:
                throughputs.append(generation_throughput)
            if tpot != 0:
                latencies.append(tpot)
        elif "schedule prefills" in line or "schedule recomputes" in line:
            result = {}
            time_str = line.split(" ")[2]
            if e2e_latency is None:
                e2e_latency = datetime.strptime(time_str, "%H:%M:%S.%f")
            result["ts"] = time_str
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            virtual_engine = int(info[0].split(":")[1].strip())
            if virtual_engine != 0:
                continue
            result["is_prefill"] = "prefills" in line
            num_seqs = int(info[1].split(":")[1].split(" ")[1])
            result["num_seqs"] = num_seqs
            result_dict.append(result)
        elif "preempt seq group" in line:
            result = {}
            time_str = line.split(" ")[2]
            result["ts"] = time_str
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            virtual_engine = int(info[0].split(":")[1].strip())
            if virtual_engine != 0:
                continue
            seq_id = info[1].split(" ")[-1]
            result["seq_id"] = seq_id
            result_dict.append(result)
        elif "Engine is gracefully shutting down" in line:
            time_str = line.split(" ")[2]
            e2e_latency = datetime.strptime(time_str, "%H:%M:%S.%f") - e2e_latency


    fields = ["ts", "kv", "compute", "throughput", "ttft", "tpot", "is_prefill", "num_seqs", "seq_id"]

    with open(f"{filename}.csv", 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(result_dict)
    
    print("Average throughput:", sum(throughputs) / len(throughputs))
    print("Average latency:", sum(latencies) / len(latencies))
    print("E2E latency:", e2e_latency)

if __name__ == "__main__":
    main(sys.argv[1])