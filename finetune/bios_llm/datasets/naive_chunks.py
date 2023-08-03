import datasets
import logging
import itertools
from typing import List
import transformers
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm

from bios_llm.utils.registry import Registry

DATASET = Registry('dataset')
logger = logging.getLogger()


class BIOSDataset(torch.utils.data.Dataset):
    def __init__(self, all_input_ids: list[int], sequence_length: int, stride: int = 1):
        super().__init__()
        self.all_input_ids = list(all_input_ids)
        self.sequence_length = sequence_length
        self.stride = stride
    
    def __len__(self):
        return int((len(self.all_input_ids) - self.sequence_length) / self.stride + 1)

    def __getitem__(self, index):
        input_ids = self.all_input_ids[index * self.stride: index * self.stride + self.sequence_length]
        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(input_ids),
        }


def create_datasets_for_bios(train_input_ids, val_input_ids, sequence_length):
    train_dataset = BIOSDataset(train_input_ids, sequence_length, stride=1)
    valid_dataset = BIOSDataset(val_input_ids, sequence_length, stride=sequence_length//2)
    logger.info('Dataset length: train=%d, val=%d', len(train_dataset), len(valid_dataset))
    logger.info('The train dataset should be roughly covered if randomly sampled %d examples.',
                len(train_input_ids) // sequence_length)
    return train_dataset, valid_dataset


@DATASET.register('dummy')
def dummy(args, **kwargs):
    train_input_ids = torch.randint(0, 100, size=(args.seq_length * 10,)).view(-1).numpy().tolist()
    val_input_ids = torch.randint(0, 100, size=(args.seq_length * 5,)).view(-1).numpy().tolist()
    sequence_length = args.seq_length
    return create_datasets_for_bios(train_input_ids, val_input_ids, sequence_length)

@DATASET.register('naive_chunks')
def naive_chunks(args, tokenizer, **kwargs):
    file = torch.load(args.naive_chunks_path)
    train, val = file['train'], file['val']
    train_input_ids = list(itertools.chain.from_iterable(dict(sorted(train.items())).values()))
    val_input_ids = list(itertools.chain.from_iterable(dict(sorted(val.items())).values()))
    logger.warning('naive_chunks_path is processed by %s; current tokenizer is %s',
                   file['tokenizer_name'], tokenizer.name_or_path)
    sequence_length = args.seq_length
    return create_datasets_for_bios(train_input_ids, val_input_ids, sequence_length)