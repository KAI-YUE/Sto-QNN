import os
import logging
import numpy as np
import pickle
import datetime

# PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# My Libraries
from deeplearning import nn_registry
from deeplearning.qnn import init_latent_params
from deeplearning.real_weights import init_weights
from deeplearning.dataset import CustomizedDataset

def load_data(config):
    with open(config.test_data_dir, "rb") as fp:
        test_data = pickle.load(fp)

    with open(config.train_data_dir, "rb") as fp:
        train_data = pickle.load(fp)

    return dict(train_data=train_data, test_data=test_data)


def init_logger(config):
    """Initialize a logger object. 
    """
    log_level = config.log_level
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    fh = logging.FileHandler(config.log_file)
    fh.setLevel(log_level)
    sh = logging.StreamHandler()
    sh.setLevel(log_level)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("-"*80)

    return logger

def save_record(config, record):
    record["cumulated_KB"].pop(0)
    current_path = os.path.dirname(__file__)
    current_time = datetime.datetime.now()
    current_time_str = datetime.datetime.strftime(current_time ,'%H_%M')
    current_time_str += str(config.delta_k)
    file_name = config.record_dir.format(current_time_str)
    with open(os.path.join(current_path, file_name), "wb") as fp:
        pickle.dump(record, fp)


def test_accuracy(model, test_dataset, device="cuda"):
    with torch.no_grad():
        dataset = CustomizedDataset(test_dataset["images"], test_dataset["labels"])
        num_samples = test_dataset["labels"].shape[0]
        accuracy = 0

        # Full Batch testing
        dividers = 100
        batch_size = int(len(dataset)/dividers)
        testing_data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        for i, samples in enumerate(testing_data_loader):
            results = model(samples["image"].to(device))
            predicted_labels = torch.argmax(results, dim=1).detach().cpu().numpy()
            accuracy += np.sum(predicted_labels == test_dataset["labels"][i*batch_size: (i+1)*batch_size]) / results.shape[0]
        
        accuracy /= dividers

    return accuracy


def train_loss(model, train_dataset, device="cuda"):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        dataset = CustomizedDataset(train_dataset["images"], train_dataset["labels"])
        loss = torch.tensor(0.)

        dividers = 100
        batch_size = int(len(dataset)/dividers)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        counter = 0
        for samples in data_loader:
            results = model(samples["image"].to(device))
            loss += criterion(results, samples["label"].to(device))
        
        loss /= dividers

    return loss.item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_qnn(config):
    # initialize the model
    sample_size = config.sample_size[0] * config.sample_size[1]
    full_model = nn_registry[config.full_model](in_dims=sample_size*config.channels, in_channels=config.channels)
    sto_qnn = nn_registry[config.model](in_dims=sample_size*config.channels, in_channels=config.channels)
    
    if os.path.exists(config.full_weight_dir):
        print("--- Load pre-trained model. ---")
        state_dict = torch.load(config.full_weight_dir)
        full_model.load_state_dict(state_dict)
    else:
        full_model.apply(init_weights)

    init_latent_params(sto_qnn, full_model)
    sto_qnn = sto_qnn.to(config.device)
    return sto_qnn

def init_full_model(config):
    # initialize the model
    sample_size = config.sample_size[0] * config.sample_size[1]
    full_model = nn_registry[config.full_model](in_dims=sample_size*config.channels, in_channels=config.channels)
    full_model.apply(init_weights)
    full_model = full_model.to(config.device)
    return full_model

def init_record(config, model):
    record = {}
    # number of trainable parameters
    record["num_parameters"] = count_parameters(model)

    # put some config info into record
    record["tau"] = config.tau
    record["batch_size"] = config.local_batch_size
    record["lr"] = config.lr
    record["predictor"] = config.predictor
    record["quantizer"] = config.quantizer
    record["scheduler"] = config.scheduler
    record["quantization_level"] = config.quantization_level

    if config.predictor == "delta_step":
        record["delta_k"] = config.delta_k

    # initialize data record 
    record["reciprocal_compress_ratio"] = []
    record["cumulated_KB"] = [0]
    record["testing_accuracy"] = []
    record["residuals"] = []
    record["quant_error"] = []
    record["loss"] = []

    return record
