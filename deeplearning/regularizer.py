import torch

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
        
        prob = 0.5*torch.tanh(module.weight) + 0.5
        loss -= lambda_*(prob*torch.log(prob) + (1-prob)*torch.log(1-prob))

    return loss 