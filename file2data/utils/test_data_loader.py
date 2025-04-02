"""
test data loader speed
usage:
python3 file2data/utils/test_data_loader.py \
    --txt_file <txt_file> \
    --batch_size <batch_size> \
    --num_workers <num_workers> \
    --prefetch_factor <prefetch_factor>
"""

from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
from tqdm import tqdm
import argparse
import os.path as osp
from loguru import logger
from torchvision import transforms
import time

class DummyDataset(Dataset):
    def __init__(self, txt_file):
        with open(txt_file, 'r') as f:
            self.data = [line.strip() for line in f.readlines()]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小到224x224
            transforms.ToTensor()           # 将PIL图像转换为Tensor
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            if osp.getsize(self.data[idx].strip()) > 1024:
                img = Image.open(self.data[idx].strip()).convert('RGB')
                return self.transform(img)
            else:
                return torch.zeros(3, 224, 224)
        except Exception as e:
            logger.warning(f'read image failed for {self.data[idx]} {e}')
            return torch.zeros(3, 224, 224, dtype=torch.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_file', type=str, default='/mnt/fileshare/workspace_robin/mnt_fileshare.txt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    args = parser.parse_args()

    dataset = DummyDataset(args.txt_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    start_time = time.time()
    count = 0
    while True:
        for batch in tqdm(dataloader, desc='loading data'):
            count += 1
            if count > 1000:
                break
        if count > 1000:
            break
    end_time = time.time()
    print(f'time cost: {end_time - start_time}s')
    sample_per_second = args.batch_size * count / (end_time - start_time)
    print(f'sample per second: {sample_per_second}')
