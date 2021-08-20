import logging
import time
from argparse import ArgumentParser

import torch
from movenet.dataset import get_dataloader
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filepath", type=str)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    torch.manual_seed(1000)

    dataloader = get_dataloader(
        args.filepath,
        input_channels=16,
        batch_size=8,
        shuffle=True,
        num_workers=args.num_workers,
    )
    n_batches = len(dataloader)
    logging.info(f"iterating through {n_batches} batches")
    writer = SummaryWriter()
    start = time.time()
    for i, (audio, video, contexts, filepaths) in enumerate(dataloader, 1):
        writer.add_scalar("n_steps", i, i)
        writer.add_scalar("percent_progress", i / n_batches, i)
        logging.info(f"[batch {i}/{n_batches}]")
    logging.info("done iterating through dataset")
