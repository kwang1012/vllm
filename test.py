# import pickle
import asyncio
import threading
import time
import multiprocessing as mp
import multiprocessing.connection
import uuid
# import torch

# def send_object(obj):
#     object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8).to(torch.device("cuda:0"))
#     size_tensor = torch.tensor([object_tensor.numel()],
#                                 dtype=torch.long, device=torch.device("cuda:0"))
#     # Send object size
#     torch.distributed.send(size_tensor,
#                             dst=1)
#     # Send object
#     torch.distributed.send(object_tensor,
#                             dst=1)

# def recv_object():
#     # Receive object size
#     size_tensor = torch.empty((1,), dtype=torch.long, device=torch.device("cuda:1"))
#     torch.distributed.recv(size_tensor, src=0)
#     object_tensor = torch.empty((size_tensor.item(),), dtype=torch.uint8, device=torch.device("cuda:1"))
#     # Receive object
#     torch.distributed.recv(object_tensor, src=0)
#     return pickle.loads(object_tensor.cpu().numpy().tobytes())

# def _run_worker_process():
#     torch.distributed.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:12345", rank=1, world_size=2)

#     avg_recv_times = []
#     for _ in range(10):
#         recv_time = time.time()
#         obj = recv_object()
#         tensor_dict = {}
#         for key, size in obj.items():
#             tensor = torch.empty(size, device=torch.device("cuda:1"), dtype=torch.float16)
#             torch.distributed.recv(tensor, src=0)
#             tensor_dict[key] = tensor
#         avg_recv_times.append(time.time() - recv_time)
    
#     print(f"Average recv time: {sum(avg_recv_times) / len(avg_recv_times)}")

# def main():
    
#     worker = mp.Process(target=_run_worker_process)
#     worker.start()
#     torch.distributed.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:12345", rank=0, world_size=2)
    
#     for _ in range(10):
#         send_object({
#             "hidden_states": (1125, 4096),
#             "residual": (1125, 4096),
#         })
#         torch.distributed.isend(torch.rand((1125, 4096), device=torch.device("cuda:0"), dtype=torch.float16), dst=1)
#         torch.distributed.isend(torch.rand((1125, 4096), device=torch.device("cuda:0"), dtype=torch.float16), dst=1)

#     worker.join()

# if __name__ == "__main__":
#     main()

tasks = {}

def _set_future_result(future, result):
    loop = future.get_loop()
    if not loop.is_closed():
        loop.call_soon_threadsafe(future.set_result, result)

def _run_worker_process(task_output: multiprocessing.connection.Connection, result_output):

    for (task_id, put_time) in iter(task_output.recv, "TERMINATE"):
        print(time.time() - put_time)
        time.sleep(0.01)
        result_output.send(task_id)

def handle_result_thread(result_output):

    for task_id in iter(result_output.recv, "TERMINATE"):
        future = tasks.pop(task_id)
        _set_future_result(future, "done")

async def enqueue_task(task_input: multiprocessing.connection.Connection):
    future = asyncio.get_running_loop().create_future()
    task_id = uuid.uuid4()
    tasks[task_id] = future
    task_input.send((task_id, time.time()))
    return await future

async def engine_step(ve, task_inputs):
    tasks = []
    for task_input in task_inputs:
        tasks.append(enqueue_task(task_input))
    
    await asyncio.gather(*tasks)

async def main():

    result_input, result_output = mp.Pipe()
    result_handler = threading.Thread(target=handle_result_thread, args=(result_input,))
    result_handler.start()
    pipes = [mp.Pipe() for _ in range(4)]
    processes = [mp.Process(target=_run_worker_process, args=(pipes[i][1], result_output)) for i in range(4)]

    for process in processes:
        process.start()

    counts = {i: 0 for i in range(4)}
    ve_tasks = [
        asyncio.create_task(engine_step(ve, [pipe[0] for pipe in pipes]))
        for ve in range(4)
    ]
    while True:
        done, _ = await asyncio.wait(
            ve_tasks,
            return_when=asyncio.FIRST_COMPLETED)
        await asyncio.sleep(0)
        for task in done:
            task.result()
            virtual_engine = ve_tasks.index(task)
            counts[virtual_engine] += 1
            if counts[virtual_engine] < 1000:
                ve_tasks[virtual_engine] = (
                    asyncio.create_task(
                        engine_step(virtual_engine, [pipe[0] for pipe in pipes])))


if __name__ == "__main__":
    asyncio.run(main())