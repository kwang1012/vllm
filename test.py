import time
import multiprocessing as mp

def _run_worker_process(queue: mp.Queue):
    while True:
        task, now = queue.get()
        print(time.time() - now)
        if task == -1:
            return
def main():
    
    task_queues = [mp.Queue() for _ in range(4)]
    
    processes = [
        mp.Process(target=_run_worker_process, args=(task_queues[i], )) for i in range(4)
    ]
    
    for p in processes:
        p.start()
    
    for i in range(1000):
        now = time.time()
        for q in task_queues:
            q.put_nowait((i, now))
        time.sleep(0.0001)
    
    for q in task_queues:
        q.put(-1)
        
    for p in processes:
        p.join()
    
if __name__ == "__main__":
    main()