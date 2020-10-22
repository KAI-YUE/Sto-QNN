import torch

def add_bnn_beta_regularizer(loss, model, lambda_=0.1):
    """add the entropy of probabilities to the loss as a regularizer.
    """
    named_modules = model.named_modules()
    next(named_modules)
    for module_name, module in named_modules:
        if not hasattr(module, "weight"):
            continue
        elif not hasattr(module.weight, "latent_param"):
            continue
        
        prob = torch.sigmoid(module.weight)
        loss += lambda_* torch.sum(prob*(1 - prob))

    return loss

def add_bnn_entropy_regularizer(loss, model, lambda_=0.1):
    """add the entropy of probabilities to the loss as a regularizer.
    """
    named_modules = model.named_modules()
    next(named_modules)
    for module_name, module in named_modules:
        if not hasattr(module, "weight"):
            continue
        elif not hasattr(module.weight, "latent_param"):
            continue
        
        prob = torch.sigmoid(module.weight)
        loss -= lambda_ * torch.sum(prob*torch.log(prob) + (1-prob)*torch.log(1-prob))

    return loss 