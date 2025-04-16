"""
speed up the process of list data
"""

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from typing import Callable, List
import os


def parallelise(
    function: Callable,
    data: List,
    verbose: bool = True,
    num_workers: int = 0,
    chunksize: int = -1,
    task_type="cpu_bound",
) -> List:
    """
    Parallelize the execution of a function over a list of data using multiprocessing.

    Args:
        function (Callable): The function to apply to each element in the data.
        data (List): The list of data to process.
        verbose (bool): Whether to display a progress bar.
        num_workers (int): The number of worker processes to use.

    Returns:
        List: The results of applying the function to the data.
    """

    if task_type == "cpu_bound":
        if num_workers == 0:
            num_workers = (
                os.cpu_count() or 1
            )  # Use all available cores by default, fallback to 1
        else:
            num_workers = max(1, num_workers)  # Ensure at least 1 worker
        if chunksize < 1:
            chunksize = min(
                max(1, len(data) // (num_workers * 4)), 50
            )  # Dynamically calculate chunksize
        with Pool(processes=num_workers) as pool:
            # Use imap_unordered for faster processing if order is not important
            results = list(
                tqdm(
                    pool.imap_unordered(function, data, chunksize),
                    total=len(data),
                    disable=not verbose,
                )
            )
    elif task_type == "io_bound":
        if num_workers == 0:
            num_workers = (os.cpu_count() or 1) * 2  # 防止None值与整数相乘
        else:
            num_workers = max(1, num_workers)
        if chunksize < 1:
            chunksize = min(max(1, len(data) // (num_workers * 4)), 50)
        with ThreadPool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(function, data, chunksize),
                    total=len(data),
                    disable=not verbose,
                )
            )
    else:
        raise ValueError(f"Invalid task_type: {task_type}")
    return results
