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
    tokens = []
    num_eviction = 0
    num_preempted_tokens = 0
    targeted_virtual_engine = 0
    batch_sizes = []
    stage_times = []
    
    print(f"============= {filename}.log =============")
    for line in lines:
        line = line.strip()
        if "Allocating" in line and not print_kv_cache:
            print(line[line.find("] ") + 2:])
            print_kv_cache = True
        elif "Loading" in line and not print_model_weights:
            print(line[line.find("] ") + 2:])
            print_model_weights = True
        elif "Avg prompt throughput" in line:
            result = {}
            time_str = line.split(" ")[2]
            result["ts"] = time_str
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            virtual_engine = int(info[0].split(":")[1].strip())
            if virtual_engine != targeted_virtual_engine:
                continue
            prompt_throughput = float(info[1].split(":")[1].strip().split(" ")[0])
            generation_throughput = float(info[2].split(":")[1].strip().split(" ")[0])
            ttft = float(info[8].split(":")[1].strip())
            tpot = float(info[9].split(":")[1].strip())
            kv_cache = float(info[6].split(":")[1].strip().rstrip("%"))
            gpu_utilization = int(info[10].split(":")[1].strip())
            batch_size = int(info[11].split(":")[1].strip())
            result["throughput"] = generation_throughput if prompt_throughput == 0 else prompt_throughput
            result["kv"] = kv_cache
            result["compute"] = gpu_utilization
            result["ttft"] = ttft
            result["tpot"] = tpot
            result["batch_size"] = batch_size
            result_dict.append(result)
            if tpot != 0:
                latencies.append(tpot)
                batch_sizes.append(batch_size)
                throughputs.append(generation_throughput)
                tokens.append(generation_throughput * tpot)
        elif "schedule prefills" in line or "schedule recomputes" in line:
            result = {}
            time_str = line.split(" ")[2]
            if e2e_latency is None:
                e2e_latency = datetime.strptime(time_str, "%H:%M:%S")
            result["ts"] = time_str
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            virtual_engine = int(info[0].split(":")[1].strip())
            if virtual_engine != targeted_virtual_engine:
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
            output_progress = int(info[2].split(" ")[-1])
            num_preempted_tokens += output_progress
            if virtual_engine != targeted_virtual_engine:
                continue
            seq_id = info[1].split(" ")[-1]
            
            result["seq_id"] = seq_id
            result_dict.append(result)
            num_eviction += 1
        elif "Engine is gracefully shutting down" in line:
            time_str = line.split(" ")[2]
            e2e_latency = datetime.strptime(time_str, "%H:%M:%S") - e2e_latency


    fields = ["ts", "kv", "compute", "throughput", "ttft", "tpot", "batch_size", "is_prefill", "num_seqs", "seq_id"]

    with open(f"{filename}.csv", 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(result_dict)
    
    # avg_throughput = sum(throughputs) / len(throughputs)
    avg_latency = sum(latencies) / len(latencies)
    overall_throughput = sum(tokens) / e2e_latency.total_seconds()
    avg_batch_size = sum(batch_sizes) / len(batch_sizes)
    print("Average batch size:", avg_batch_size)
    print("Overall throughput:", overall_throughput)
    print("Average TPOT:", avg_latency)
    print(f"Total evictions: {num_eviction}, Evicted output tokens: {num_preempted_tokens}")
    print("E2E latency:", e2e_latency)

if __name__ == "__main__":
    main(sys.argv[1])