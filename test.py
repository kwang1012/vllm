import asyncio
import time

async def run_with_lock(fn, lock):
    async with lock:
        await fn()

async def foo():
    return

async def task(fn, lock, input_queue, output_queue, terminate_flag: asyncio.Event):
    while not terminate_flag.is_set():
        input = await input_queue.get()
        await run_with_lock(fn, lock)
        await output_queue.put(input)

async def main():
    locks = [asyncio.Lock() for _ in range(4)]
    start = time.time()
    tasks = []
    per_times = []
    request_queues = [asyncio.Queue(1000) for _ in range(4)]
    output_queues = [asyncio.Queue(1000) for _ in range(4)]
    terminate_flag = asyncio.Event()
    task_pools = [asyncio.create_task(task(foo, locks[i], request_queues[i], output_queues[i], terminate_flag)) for i in range(4)]
    for n in range(1000):
        per_time = time.time()
        for i in range(4):
            request_queues[i].put_nowait(n)
        
        for i in range(4):
            await output_queues[i].get()
            # tasks.append(asyncio.create_task(run_with_lock(foo, locks[i])))
        # asyncio.gather(*tasks)
        per_time = time.time() - per_time
        per_times.append(per_time)
    terminate_flag.set()
    print("Avg time:", sum(per_times) / len(per_times))
    
    start = time.time()
    tasks = []
    for _ in range(4):
        await run_with_lock(foo, locks[i])

    print(time.time() - start)

if __name__ == "__main__":
    asyncio.run(main())