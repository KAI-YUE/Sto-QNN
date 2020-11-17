import numpy as np

# PyTorch Libraries
import torch
from torch.utils.data import DataLoader

# My Libraries
from config import *
from deeplearning.initialize import *
from deeplearning import CustomizedDataset
from deeplearning.regularizer import *

def train_qnn(model, config, logger, record):
    device = config.device
    dataset = load_data(config)
    dataset_type = parse_dataset_type(config)
    train_dataloader = DataLoader(CustomizedDataset(dataset["train_data"]["images"], 
                                    dataset["train_data"]["labels"], 
                                    dataset_type), batch_size=config.batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.latent_parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # optimizer = torch.optim.SGD(model.latent_parameters(), lr=config.lr)

    # before optimization, report the result first
    with torch.no_grad():
        # validate the model and log test accuracy
        loss = train_loss(model, dataset["train_data"], dataset_type, device=config.device)
        test_acc = test_accuracy(model, dataset["test_data"], dataset_type, device=config.device)

        logger.info("Test accuracy {:.4f}".format(test_acc))
        logger.info("Train loss {:.4f}".format(loss))
        logger.info("")

    for epoch in range(config.total_epoch):
        logger.info("--- Epoch {:d} ---".format(epoch))
        for iteration, sample in enumerate(train_dataloader):
            image = sample["image"].to(device)
            label = sample["label"].to(device)

            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

            # record the test accuracy
            if iteration % config.log_iters == 0:
                with torch.no_grad():
                    # loss = train_loss(model, dataset["train_data"], device=config.device)
                    test_acc = test_accuracy(model, dataset["test_data"], dataset_type, config.device)
                    sampled_qnn_acc = test_qnn_accuracy(model, dataset["test_data"], config.device, config)
                
                record["testing_accuracy"].append(test_acc)
                logger.info("Train loss {:.4f}".format(loss))
                logger.info("Test accuracy {:.4f}".format(test_acc))
                logger.info("Sampled QNN Test accuracy {:.4f}".format(sampled_qnn_acc))

    torch.save(model.latent_param_dict(), "tnn.pth")

def train_bnn(model, config, logger, record):
    device = config.device
    dataset = load_data(config)
    dataset_type = parse_dataset_type(config)
    train_dataloader = DataLoader(CustomizedDataset(dataset["train_data"]["images"], 
                                    dataset["train_data"]["labels"], 
                                    dataset_type), batch_size=config.batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.latent_parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

    # before optimization, report the result first 
    with torch.no_grad():
        # validate the model and log test accuracy
        loss = train_loss(model, dataset["train_data"], dataset_type, device=config.device)
        test_acc = test_accuracy(model, dataset["test_data"], dataset_type, device=config.device)
        
        logger.info("Test accuracy {:.4f}".format(test_acc))
        logger.info("Train loss {:.4f}".format(loss))
        logger.info("")

    for epoch in range(config.total_epoch):
        logger.info("--- Epoch {:d} ---".format(epoch))
        for iteration, sample in enumerate(train_dataloader):
            image = sample["image"].to(device)
            label = sample["label"].to(device)

            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, label)
            # loss = add_bnn_entropy_regularizer(loss, model, lambda_=config.lambda_)
            # loss = add_bnn_beta_regularizer(loss, model, lambda_=config.lambda_)

            loss.backward()
            optimizer.step()

            # record the test accuracy
            if iteration % config.log_iters == 0:
                with torch.no_grad():
                    # loss = train_loss(model, dataset["train_data"], device=config.device)
                    test_acc = test_accuracy(model, dataset["test_data"], dataset_type, device=config.device)
                    sampled_bnn_acc = test_bnn_accuracy(model, dataset["test_data"], config.device, config, logger)

                record["testing_accuracy"].append(test_acc)
                record["sampled_accuracy"].append(sampled_bnn_acc)
                logger.info("Train loss {:.4f}".format(loss))
                logger.info("Test accuracy {:.4f}".format(test_acc))
                logger.info("Sampled BNN accuracy {:.4f}".format(sampled_bnn_acc))
        
        if epoch == 10:
            optimizer = torch.optim.Adam(model.latent_parameters(), lr=0.1*config.lr, weight_decay=config.weight_decay)

    torch.save(model.latent_param_dict(), "bnn.pth")

def train_full_model(model, config, logger, record):
    device = config.device
    dataset = load_data(config) 
    dataset_type = parse_dataset_type(config)
    train_dataloader = DataLoader(CustomizedDataset(dataset["train_data"]["images"], 
                                    dataset["train_data"]["labels"], 
                                    dataset_type), batch_size=config.batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # before optimization, report the result first 
    with torch.no_grad():
        # validate the model and log test accuracy
        loss = train_loss(model, dataset["train_data"], dataset_type, device=config.device)
        test_acc = test_accuracy(model, dataset["test_data"], dataset_type, device=config.device)

        logger.info("Test accuracy {:.4f}".format(test_acc))
        logger.info("Train loss {:.4f}".format(loss))
        logger.info("")

    for epoch in range(config.total_epoch):
        logger.info("--- Epoch {:d} ---".format(epoch))
        for iteration, sample in enumerate(train_dataloader):
            image = sample["image"].to(device)
            label = sample["label"].to(device)

            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

            # record the test accuracy
            if iteration % config.log_iters == 0:
                with torch.no_grad():
                    # loss = train_loss(model, dataset["train_data"], dataset_type, device=config.device)
                    test_acc = test_accuracy(model, dataset["test_data"], dataset_type, device=config.device)

                record["testing_accuracy"].append(test_acc)
                logger.info("Train loss {:.4f}".format(loss))
                logger.info("Test accuracy {:.4f}".format(test_acc))

        if epoch == 10:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1*config.lr, weight_decay=config.weight_decay)

    torch.save(model.state_dict(), "full_model.pth")

def main():
    config = load_config()
    logger = init_logger(config)
    record = init_record(config)

    if config.device == "cuda":
        torch.backends.cudnn.benchmark = True

    # ----------------------------
    # mode 0: train the quantized neural network
    # mode 1: train the binary neural network
    # mode 2: train the full precision neural network
    #-----------------------------

    if config.mode == 0:
        qnn = init_qnn(config, logger)
        train_qnn(qnn, config, logger, record)
    elif config.mode == 1:
        bnn = init_bnn(config, logger)

        # bnn_latent_param = torch.load("/media/kaiyue/2D8A97B87FB4A806/Datasets/tmodels/vote10.pth")
        # bnn_latent_param = torch.load("/media/kaiyue/2D8A97B87FB4A806/Datasets/mmodels/bnn_modified.pth")
        # bnn.load_latent_param_dict(bnn_latent_param)

        train_bnn(bnn, config, logger, record)
    elif config.mode == 2:
        model = init_full_model(config, logger)
        train_full_model(model, config, logger, record)

    save_record(config, record)
    
if __name__ == '__main__':
    main()
