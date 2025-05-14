import argparse
import os
import pathlib import Path 
import time
import tiktoken
import torch

from gpt.gpt import GPTModel, create_dataloader_v1, generate_text_simple
from gpt.gpt_train import calc_loss_batch, evaluate_model, plot_losses

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data = file.read()
    return text_data


def create_dataloader(text_data, train_ratio, batch_size, max_length, stride, num_workers =0):
    split_idx = int(train_ratio * len(text_data))
    