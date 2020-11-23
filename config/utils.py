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
from deeplearning.dataset import CustomizedDataset

def load_data(config):
    with open(config.test_data_dir, "rb") as fp:
        test_data = pickle.load(fp)

    with open(config.train_data_dir, "rb") as fp:
        train_data = pickle.load(fp)

    # with open("/media/kaiyue/2D8A97B87FB4A806/Datasets/user_with_data/iid_map.dat", "rb") as fp:
    #     user_dataidx_map = pickle.load(fp)

    # user_id = config.user_id
    # train_data["images"] = train_data["images"][user_dataidx_map[user_id]]
    # train_data["labels"] = train_data["labels"][user_dataidx_map[user_id]]

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
    current_path = os.path.dirname(__file__)
    current_time = datetime.datetime.now()
    current_time_str = datetime.datetime.strftime(current_time ,'%H_%M')
    file_name = config.record_dir.format(current_time_str)
    with open(os.path.join(current_path, file_name), "wb") as fp:
        pickle.dump(record, fp)

def test_accuracy(model, test_dataset, type_, device="cuda"):
    with torch.no_grad():
        model.eval()
        dataset = CustomizedDataset(test_dataset["images"], test_dataset["labels"], type_)
        num_samples = test_dataset["labels"].shape[0]
        accuracy = 0

        dividers = 100
        batch_size = int(len(dataset)/dividers)
        testing_data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        for i, samples in enumerate(testing_data_loader):
            results = model(samples["image"].to(device))
            predicted_labels = torch.argmax(results, dim=1).detach().cpu().numpy()
            accuracy += np.sum(predicted_labels == test_dataset["labels"][i*batch_size: (i+1)*batch_size]) / results.shape[0]
        
        accuracy /= dividers

        model.train()

    return accuracy

def test_qnn_accuracy(qnn_model, test_dataset, device, config):
    """test the accuracy of a sampled qnn from the latent distribution.
    """
    with torch.no_grad():
        qnn_model.eval()
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
            
            # mle sampling
            # sampled_qnn_state_dict[module_name + ".weight"] = torch.argmax(prob, dim=-1) - 1 
            
            # random sampling
            random_variable = torch.rand_like(prob[..., 0])
            sampled_weight = torch.zeros_like(prob[..., 0])
            sampled_weight = torch.where(random_variable < prob[..., 0], torch.tensor(-1.), sampled_weight)
            sampled_weight = torch.where(1 - prob[...,2] < random_variable, torch.tensor(1.), sampled_weight)
            sampled_qnn_state_dict[module_name + ".weight"] = sampled_weight.clone()

        sampled_qnn.load_state_dict(sampled_qnn_state_dict)
        sampled_qnn = sampled_qnn.to(device)
        dataset_type = parse_dataset_type(config)
        acc = test_accuracy(sampled_qnn, test_dataset, dataset_type, device)

        qnn_model.train()
    return acc

def test_bnn_accuracy(bnn_model, test_dataset, device, config, logger):
    with torch.no_grad():
        bnn_model.eval()
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

            sampled_bnn_state_dict[module_name + ".weight"] = module.weight.latent_param.sign()
            prob = torch.sigmoid(module.weight)
            # entropy -= torch.sum(prob*torch.log(prob) + (1-prob)*torch.log(1-prob)) 
            
            # rand_variable = torch.rand_like(module.weight.latent_param)
            # prob_equal_one = torch.sigmoid(module.weight.latent_param)
            # ones_tensor = torch.ones_like(module.weight.latent_param)
            # zeros_tensor = torch.zeros_like(module.weight.latent_param)
            # sampled_bnn_state_dict[module_name + ".weight"] = torch.where(rand_variable < prob_equal_one, ones_tensor, -ones_tensor)

        # logger.info("Entropy: {:.3f}".format(entropy))

        sampled_bnn.load_state_dict(sampled_bnn_state_dict)
        sampled_bnn = sampled_bnn.to(device)
        dataset_type = parse_dataset_type(config)
        acc = test_accuracy(sampled_bnn, test_dataset, dataset_type, device)

        bnn_model.train()

    return acc            

def train_loss(model, train_dataset, type_, device="cuda"):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        dataset = CustomizedDataset(train_dataset["images"], train_dataset["labels"], type_)
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

def init_record(config):
    record = {}

    # put some config info into record
    record["batch_size"] = config.batch_size
    record["lr"] = config.lr

    # initialize data record 
    record["testing_accuracy"] = []
    record["sampled_accuracy"] = []
    record["loss"] = []

    return record
