import numpy as np

# PyTorch Libraries
import torch
from torch.utils.data import DataLoader

# My Libraries
from config import *
from deeplearning import CustomizedDataset

def train(model, config, logger):
    device = config.device
    dataset = load_data(config) 
    train_dataloader = DataLoader(CustomizedDataset(dataset["train_data"]["images"],dataset["train_data"]["labels"]),
                                  batch_size=config.batch_size)
    
    criterion = nn.CrossEntropyLoss()
    model.latent_parameters()
    optimizer = torch.optim.Adam(model.latent_parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # before optimization, report the result first 
    with torch.no_grad():
        # validate the model and log test accuracy
        loss = train_loss(model, dataset["train_data"], device=config.device)
        test_acc = test_accuracy(model, dataset["test_data"], device=config.device)

        logger.info("Test accuracy {:.4f}".format(test_acc))
        logger.info("Train loss {:.4f}".format(loss))
        logger.info("")

    for epoch in range(config.total_epoch):
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
                    test_acc = test_accuracy(model, dataset["test_data"], device=config.device)
                
                logger.info("Test accuracy {:.4f}".format(test_acc))

def main():
    config = load_config()
    logger = init_logger(config)
    model = init_model(config)
    train(model, config, logger)
    
if __name__ == '__main__':
    main()
