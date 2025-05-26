"""
实现一个玩具级别的gpu，能够在多张GPU上进行计算，要求占用显存小，但GPU利用率高
其中运行的GPU编号可以指定，也可以自动选择
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from tqdm import trange
import datetime
import socket


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
        )

    def forward(self, x):
        return self.net(x)


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]


def setup(rank, world_size, timeout):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(get_free_port()))
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=timeout))


def train(rank, world_size, epochs, batch_size, timeout):
    setup(rank, world_size, timeout)

    # 创建模型并移动到对应的GPU
    model = ToyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 创建随机数据
    x = torch.randn(batch_size, 1000).to(rank)
    y = torch.randn(batch_size, 500).to(rank)

    # 训练循环
    if epochs <= 0:
        # run forever
        epoch = 0
        init_time = time.time()
        while True:
            start_time = time.time()
            output = model(x)
            loss = nn.MSELoss()(output, y)
            optimizer.zero_grad()
            loss.backward()
            torch.distributed.barrier()
            optimizer.step()

            epoch += 1
            if rank == 0:
                duration = time.time() - init_time
                if duration > 10:
                    init_time = time.time()
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Time: {time.time() - start_time:.2f}s, Duration: {duration:.2f}s")

            time.sleep(0.01)
    else:
        tbar = trange(epochs)
        for epoch in tbar:
            start_time = time.time()
            output = model(x)
            loss = nn.MSELoss()(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Time: {time.time() - start_time:.2f}s")
                tbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Time: {time.time() - start_time:.2f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=[], nargs="+", help="指定运行的GPU编号")
    parser.add_argument("--batch-size", type=int, default=8192, help="批量大小")
    parser.add_argument("--epochs", type=int, default=0, help="训练轮数")
    parser.add_argument("--timeout", type=int, default=10, help="同步超时时间")
    args = parser.parse_args()

    if len(args.gpus) == 0:
        world_size = torch.cuda.device_count()
        if world_size == 0:
            print("没有可用的GPU")
            return
        args.gpus = list(range(world_size))
    else:
        world_size = len(args.gpus)

    # 设置可见的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))
    print(f"使用 {world_size} 个GPU进行训练")
    mp.spawn(
        train,
        args=(world_size, args.epochs, args.batch_size, args.timeout),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
