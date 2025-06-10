"""
实现一个玩具级别的gpu，能够在多张GPU上进行计算，要求占用显存小，但GPU利用率高
其中运行的GPU编号可以指定，也可以自动选择

特色功能:
1. 当指定时间窗口内(默认1min内， 每秒检查一次)系统的显存平均使用率超过指定值(默认为30%时), 则自动停止训练。反之则继续训练
2. 对每块GPU单独进行监控，当某块GPU的显存使用率超过指定值时，则自动停止训练。反之则继续训练
3. 当终止主进程时，所有子进程也会自动停止
4. 每隔一个时间窗口，以表格的形式，打印一次平均的GPU显存使用率 及 GPU利用率
5. 当GPU显存低于指定值时，自动恢复训练
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
import threading
from collections import deque
import numpy as np
import signal
import sys
from tabulate import tabulate
import pynvml


class GPUMonitor:
    def __init__(self, window_size=60, threshold=30, recovery_threshold=20, gpu_ids=None):
        """
        初始化GPU监控器
        Args:
            window_size: 时间窗口大小（秒）
            threshold: 显存使用率阈值（百分比）
            recovery_threshold: 恢复训练的显存使用率阈值（百分比）
            gpu_ids: 要监控的GPU ID列表
        """
        self.window_size = window_size
        self.threshold = threshold
        self.recovery_threshold = recovery_threshold
        self.gpu_ids = gpu_ids or list(range(torch.cuda.device_count()))
        self.memory_usage = {gpu_id: deque(maxlen=window_size) for gpu_id in self.gpu_ids}
        self.gpu_utilization = {gpu_id: deque(maxlen=window_size) for gpu_id in self.gpu_ids}
        self.should_stop = False
        self.monitor_thread = None
        self.last_print_time = time.time()
        self.last_status = None  # 用于跟踪状态变化
        
        # 初始化NVML
        try:
            pynvml.nvmlInit()
            self.nvml_initialized = True
        except pynvml.NVMLError:
            print("警告：无法初始化NVML，将使用备用方法获取GPU信息")
            self.nvml_initialized = False
        
    def get_gpu_memory_usage(self, gpu_id):
        """获取指定GPU的显存使用率"""
        if self.nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return (info.used / info.total) * 100
            except pynvml.NVMLError:
                pass
        
        # 备用方法
        with torch.cuda.device(gpu_id):
            return torch.cuda.memory_allocated() / torch.cuda.get_device_properties(gpu_id).total_memory * 100
    
    def get_gpu_utilization(self, gpu_id):
        """获取指定GPU的利用率"""
        if self.nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return info.gpu
            except pynvml.NVMLError:
                pass
        
        # 备用方法：使用nvidia-smi
        try:
            import subprocess
            result = subprocess.check_output(
                f'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i {gpu_id}',
                shell=True
            ).decode().strip()
            return float(result)
        except:
            return 0.0
        
    def print_gpu_stats(self):
        """打印GPU统计信息表格"""
        headers = ['GPU ID', '显存使用率(%)', 'GPU利用率(%)']
        table_data = []
        
        for gpu_id in self.gpu_ids:
            if len(self.memory_usage[gpu_id]) > 0:
                avg_memory = np.mean(self.memory_usage[gpu_id])
                avg_util = np.mean(self.gpu_utilization[gpu_id])
                table_data.append([gpu_id, f"{avg_memory:.1f}", f"{avg_util:.1f}"])
        
        print("\nGPU使用情况统计:")
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
    def monitor_loop(self):
        """监控循环"""
        while not self.should_stop:
            current_time = time.time()
            
            for gpu_id in self.gpu_ids:
                # 获取显存使用率
                memory_usage = self.get_gpu_memory_usage(gpu_id)
                self.memory_usage[gpu_id].append(memory_usage)
                
                # 获取GPU利用率
                gpu_util = self.get_gpu_utilization(gpu_id)
                self.gpu_utilization[gpu_id].append(gpu_util)
                
                # 检查是否超过阈值
                if len(self.memory_usage[gpu_id]) == self.window_size:
                    avg_usage = np.mean(self.memory_usage[gpu_id])
                    if avg_usage > self.threshold:
                        if self.last_status != 'stopped':
                            print(f"\n警告：GPU {gpu_id} 过去{self.window_size}秒内平均显存使用率({avg_usage:.1f}%)超过阈值({self.threshold}%)")
                            self.last_status = 'stopped'
                        self.should_stop = True
                    elif avg_usage < self.recovery_threshold:
                        if self.last_status != 'running':
                            print(f"\n提示：GPU {gpu_id} 显存使用率({avg_usage:.1f}%)已低于恢复阈值({self.recovery_threshold}%)")
                            self.last_status = 'running'
                        self.should_stop = False
            
            # 定期打印统计信息
            if current_time - self.last_print_time >= self.window_size:
                self.print_gpu_stats()
                self.last_print_time = current_time
            
            time.sleep(1)
    
    def start(self):
        """启动监控"""
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self):
        """停止监控"""
        self.should_stop = True
        if self.monitor_thread:
            self.monitor_thread.join()
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

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


def setup(rank, world_size, timeout, master_addr, master_port):
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=timeout))


def train(rank, world_size, epochs, batch_size, timeout, master_addr, master_port, monitor_window=60, monitor_threshold=30, monitor_recovery_threshold=20):
    # 设置信号处理
    def signal_handler(signum, frame):
        print(f"\n进程 {rank} 收到终止信号，正在停止...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    setup(rank, world_size, timeout, master_addr, master_port)

    # 创建模型并移动到对应的GPU
    model = ToyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 创建随机数据
    x = torch.randn(batch_size, 1000).to(rank)
    y = torch.randn(batch_size, 500).to(rank)

    # 初始化GPU监控器
    monitor = GPUMonitor(
        window_size=monitor_window,
        threshold=monitor_threshold,
        recovery_threshold=monitor_recovery_threshold,
        gpu_ids=[rank]
    )
    monitor.start()

    on_training = True
    try:
        # 训练循环
        if epochs <= 0:
            # run forever
            epoch = 0
            init_time = time.time()
            while True:
                start_time = time.time()
                if monitor.should_stop:
                    if on_training:
                        print(f"\nGPU {rank} 暂停训练")
                        on_training = False
                    time.sleep(1)  # 暂停时降低检查频率
                    continue

                if not on_training:
                    print(f"\nGPU {rank} 恢复训练")
                    on_training = True

                output = model(x)
                loss = nn.MSELoss()(output, y)
                optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                epoch += 1
                duration = time.time() - init_time
                if duration > 10:
                    init_time = time.time()
                    if rank == 0:
                        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Time: {time.time() - start_time:.2f}s, Duration: {duration:.2f}s")
                    time.sleep(0.01)

                if epoch > 1024 * 1024 * 1024:
                        epoch = 0
        else:
            tbar = trange(epochs)
            for epoch in tbar:
                start_time = time.time()
                if monitor.should_stop:
                    if on_training:
                        print(f"\nGPU {rank} 暂停训练")
                        on_training = False
                    time.sleep(1)  # 暂停时降低检查频率
                    continue

                if not on_training:
                    print(f"\nGPU {rank} 恢复训练")
                    on_training = True

                output = model(x)
                loss = nn.MSELoss()(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if rank == 0:
                    tbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Time: {time.time() - start_time:.2f}s")
    finally:
        monitor.stop()
        # 确保所有进程同步退出
        torch.distributed.barrier()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=[], nargs="+", help="指定运行的GPU编号")
    parser.add_argument("--batch-size", type=int, default=256, help="批量大小")
    parser.add_argument("--epochs", type=int, default=0, help="训练轮数， 0代表无限循环")
    parser.add_argument("--timeout", type=int, default=10, help="同步超时时间(秒), 当kill main程序时, 其它程序会超时死亡")
    parser.add_argument("--master-port", type=int, default=get_free_port(), help="主节点端口")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1", help="主节点地址")
    parser.add_argument("--monitor-window", type=int, default=60, help="显存监控时间窗口（秒）")
    parser.add_argument("--monitor-threshold", type=float, default=30, help="显存使用率阈值（百分比）")
    parser.add_argument("--monitor-recovery-threshold", type=float, default=20, help="恢复训练的显存使用率阈值（百分比）")
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

    # 设置进程启动方法
    mp.set_start_method('spawn', force=True)

    try:
        mp.spawn(
            train,
            args=(world_size, args.epochs, args.batch_size, args.timeout, args.master_addr, args.master_port,
                  args.monitor_window, args.monitor_threshold, args.monitor_recovery_threshold),
            nprocs=world_size,
            join=True,
        )
    except KeyboardInterrupt:
        print("\n收到终止信号，正在停止所有进程...")
        sys.exit(0)

if __name__ == "__main__":
    main()
