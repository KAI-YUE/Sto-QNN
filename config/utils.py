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
from deeplearning.qnn import init_latent_params, init_bnn_params
from deeplearning.real_weights import init_weights
from deeplearning.dataset import CustomizedDataset

def load_data(config):
    with open(config.test_data_dir, "rb") as fp:
        test_data = pickle.load(fp)

    with open(config.train_data_dir, "rb") as fp:
        train_data = pickle.load(fp)

    return dict(train_data=train_data, test_data=test_data)

def parse_dataset_type(config):
    if "fmnist" in config.train_data_dir:
        type_ = "fmnist"
    elif "mnist" in config.train_data_dir:
        type_ = "mnist"
    elif "cifar" in config.train_data_dir:
        type_ = "cifar"
    
    return type_

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

def test_accuracy(qnn_model, test_dataset, device="cuda"):
    with torch.no_grad():
        dataset = CustomizedDataset(test_dataset["images"], test_dataset["labels"])
        num_samples = test_dataset["labels"].shape[0]
        accuracy = 0

        dividers = 100
        batch_size = int(len(dataset)/dividers)
        testing_data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        for i, samples in enumerate(testing_data_loader):
            results = qnn_model(samples["image"].to(device))
            predicted_labels = torch.argmax(results, dim=1).detach().cpu().numpy()
            accuracy += np.sum(predicted_labels == test_dataset["labels"][i*batch_size: (i+1)*batch_size]) / results.shape[0]
        
        accuracy /= dividers

    return accuracy

def test_qnn_accuracy(qnn_model, test_dataset, device, config):
    """test the accuracy of a sampled qnn from the latent distribution.
    """
    with torch.no_grad():
        sample_size = config.sample_size[0] * config.sample_size[1]
        sampled_qnn = nn_registry[config.full_model](in_dims=sample_size*config.channels, in_channels=config.channels)
        sampled_qnn.load_state_dict(qnn_model.state_dict())
        
        sampled_qnn_state_dict = sampled_qnn.state_dict()

        named_modules = qnn_model.named_modules()
        next(named_modules)
        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue
            
            # probability vector [pr(w_b=-1), pr(w_b=0), pr(w_b=1)] 
            theta = torch.sigmoid(module.weight.latent_param)
            theta_shape = torch.tensor(theta.shape).tolist()
            theta_shape[-1] = 3
            prob = torch.zeros(theta_shape)
            prob[..., 0] = theta[..., 0]
            prob[..., 1] = 1 - theta[..., 0] - theta[..., 1] 
            prob[..., 2] = theta[..., 1]

            sampled_qnn_state_dict[module_name + ".weight"] = torch.argmax(prob, dim=-1) - 1 
        
        sampled_qnn.load_state_dict(sampled_qnn_state_dict)
        sampled_qnn = sampled_qnn.to(device)
        acc = test_accuracy(sampled_qnn, test_dataset, device)
    
    return acc

def test_bnn_accuracy(bnn_model, test_dataset, device, config, logger):
    with torch.no_grad():
        sample_size = config.sample_size[0] * config.sample_size[1]
        sampled_bnn = nn_registry[config.full_model](in_dims=sample_size*config.channels, in_channels=config.channels)
        sampled_bnn.load_state_dict(bnn_model.state_dict())
        
        sampled_bnn_state_dict = sampled_bnn.state_dict()

        entropy = 0

        named_modules = bnn_model.named_modules()
        next(named_modules)
        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue

            sampled_bnn_state_dict[module_name + ".weight"] = module.weight.sign()
            prob = torch.sigmoid(module.weight)
            entropy -= torch.sum(prob*torch.log(prob) + (1-prob)*torch.log(1-prob)) 
            
            # rand_variable = torch.rand_like(module.weight)
            # prob_equal_one = torch.sigmoid(module.weight)
            # ones_tensor = torch.ones_like(module.weight)
            # zeros_tensor = torch.zeros_like(module.weight)
            # sampled_bnn_state_dict[module_name + ".weight"] = torch.where(rand_variable < prob_equal_one, ones_tensor, -ones_tensor)

        logger.info("Entropy: {:.3f}".format(entropy))

        sampled_bnn.load_state_dict(sampled_bnn_state_dict)
        sampled_bnn = sampled_bnn.to(device)
        acc = test_accuracy(sampled_bnn, test_dataset, device)
    
    return acc            

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

def count_parameters(qnn_model):
    return sum(p.numel() for p in qnn_model.parameters() if p.requires_grad)

def init_qnn(config, logger):
    # initialize the qnn_model
    sample_size = config.sample_size[0] * config.sample_size[1]
    full_model = nn_registry[config.full_model](in_dims=sample_size*config.channels, in_channels=config.channels)
    sto_qnn = nn_registry[config.qnn_model](in_dims=sample_size*config.channels, in_channels=config.channels)
    
    if os.path.exists(config.full_weight_dir):
        logger.info("--- Load pre-trained full precision model. ---")
        state_dict = torch.load(config.full_weight_dir)
        full_model.load_state_dict(state_dict)
    else:
        logger.info("--- Train quantized qnn_model from scratch. ---")
        full_model.apply(init_weights)

    init_latent_params(sto_qnn, full_model)
    sto_qnn = sto_qnn.to(config.device)
    return sto_qnn

def init_bnn(config, logger):
    # initialize the bnn_model
    sample_size = config.sample_size[0] * config.sample_size[1]
    full_model = nn_registry[config.full_model](in_dims=sample_size*config.channels, in_channels=config.channels)
    sto_bnn = nn_registry[config.qnn_model](in_dims=sample_size*config.channels, in_channels=config.channels)
    
    if os.path.exists(config.full_weight_dir):
        logger.info("--- Load pre-trained full precision model. ---")
        state_dict = torch.load(config.full_weight_dir)
        full_model.load_state_dict(state_dict)
    else:
        logger.info("--- Train quantized qnn_model from scratch. ---")
        full_model.apply(init_weights)

    init_bnn_params(sto_bnn, full_model)
    sto_bnn = sto_bnn.to(config.device)
    return sto_bnn

def init_full_model(config, logger):
    # initialize the qnn_model
    logger.info("--- Train full precision qnn_model from scratch. ---")
    sample_size = config.sample_size[0] * config.sample_size[1]
    full_model = nn_registry[config.full_model](in_dims=sample_size*config.channels, in_channels=config.channels)
    full_model.apply(init_weights)
    full_model = full_model.to(config.device)
    return full_model

def init_record(config, qnn_model):
    record = {}
    # number of trainable parameters
    record["num_parameters"] = count_parameters(qnn_model)

    # put some config info into record
    record["tau"] = config.tau
    record["batch_size"] = config.local_batch_size
    record["lr"] = config.lr

    if config.predictor == "delta_step":
        record["delta_k"] = config.delta_k

    # initialize data record 
    record["testing_accuracy"] = []
    record["residuals"] = []
    record["quant_error"] = []
    record["loss"] = []

    return record
