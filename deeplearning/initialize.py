import os

# PyTorch libraries
import torch
import torch.nn as nn

# My libraries
from deeplearning import nn_registry

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

    init_tnn_latent_params(sto_qnn, full_model)
    sto_qnn.freeze_weight()
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
        logger.info("--- Train quantized bnn_model from scratch. ---")
        full_model.apply(init_weights)

    init_bnn_latent_params(sto_bnn, full_model)

    if config.freeze_fc:
        sto_bnn.freeze_final_layer()

    sto_bnn = sto_bnn.to(config.device)
    return sto_bnn

def init_full_model(config, logger):
    # initialize the qnn_model
    logger.info("--- Train full precision model from scratch. ---")
    sample_size = config.sample_size[0] * config.sample_size[1]
    full_model = nn_registry[config.full_model](in_dims=sample_size*config.channels, in_channels=config.channels)
    full_model.apply(init_weights)

    if config.freeze_fc:
        full_model.freeze_final_layer()

    full_model = full_model.to(config.device)
    return full_model

# initialization method reference 
# Roth, Wolfgang, Günther Schindler, Holger Fröning, and Franz Pernkopf. 
# "Training Discrete-Valued Neural Networks with Sign Activations Using Weight Distributions." 
# In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 382-398. Springer, Cham, 2019.

def init_tnn_latent_params(model, ref_model, **kwargs):
    """Initialize the multinomial distribution parameters. 
    theta_0 = Pr(w=-1) 
    theta_1 = Pr(w=1)

    Args:
        ref_model (StoQNN):     the reference floating point model
    """
    if "p_max" in kwargs:
        p_max = torch.tensor(kwargs["p_max"])
    else:
        p_max = torch.tensor(0.95)
    if "p_min" in kwargs:
        p_min = torch.tensor(kwargs["p_min"])
    else:
        p_min = torch.tensor(0.05)
    if "method" in kwargs:
        method = kwargs["method"]
    else:
        method = "probability" 
    

    ref_state_dict = ref_model.state_dict()
    model.load_state_dict(ref_state_dict)
    named_modules = model.named_modules()
    next(named_modules)

    if method == "probability":
        delta = p_max - p_min/2
        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue
            
            # pr(w_q = -1)
            ref_w = ref_state_dict[module_name + ".weight"]
            ref_w = ref_w/ref_w.std()

            idx = torch.logical_and(ref_w >-1, ref_w<=0)
            prob_m1 = torch.where(ref_w <= -1, p_max, ref_w)
            prob_m1 = torch.where(ref_w > 0, p_min/2, prob_m1)
            prob_m1[idx] = p_max - delta*(ref_w[idx] + 1)

            # pr(w_q = 1)
            idx = torch.logical_and(ref_w >0, ref_w<=1)
            prob_p1 = torch.where(ref_w > 1, p_max, ref_w)
            prob_p1 = torch.where(ref_w <= 0, p_min/2, prob_p1)
            prob_p1[idx] = p_min/2 + delta*ref_w[idx]

            # take the logits of the probability, i.e., log(y/(1-y)) = log(-1+1/(1-y))
            module.weight.latent_param.data[..., 0] = torch.log(-1+1/(1-prob_m1))
            module.weight.latent_param.data[..., 1] = torch.log(-1+1/(1-prob_p1))

    elif method == "shayer":
        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue
                
            # normalize the weight
            ref_w = ref_state_dict[module_name + ".weight"]
            # normalized_w = ref_w / ref_w.std()
            # abs_normalized_w = normalized_w.abs()
            # normalized_w = 2*(ref_w - ref_w.min())/(ref_w.max() - ref_w.min()) - 1
            normalized_w = ref_w.clamp(-0.99, 0.99)

            # take the logits of the probability, i.e., log(y/(1-y)) = log(-1+1/(1-y))
            prob_m1 = p_min - p_max*normalized_w
            prob_m1 = torch.clamp(prob_m1, p_min, p_max)
            prob_p1 = p_min  + p_max*normalized_w
            prob_p1 = torch.clamp(prob_p1 , p_min, p_max)

            module.weight.latent_param.data[..., 0] = torch.log(-1+1/(1-prob_m1))
            module.weight.latent_param.data[..., 1] = torch.log(-1+1/(1-prob_p1))


def init_bnn_latent_params(model, ref_model, **kwargs):
    """Initialize the Bernoulli distribution parameters. 
    Pr(w_b=1) = 0.5*(1 + w) 

    Args:
        ref_model (nn.Module):     the reference floating point model
    """
    if "p_max" in kwargs:
        p_max = torch.tensor(kwargs["p_max"])
    else:
        p_max = torch.tensor(0.95)
    if "p_min" in kwargs:
        p_min = torch.tensor(kwargs["p_min"])
    else:
        p_min = torch.tensor(0.05)
    if "method" in kwargs:
        method = kwargs["method"]
    else:
        # method = "plain"
        method = "probability"
        # method = "test" 

    ref_state_dict = ref_model.state_dict()
    model.load_state_dict(ref_state_dict)
    named_modules = model.named_modules()
    next(named_modules)

    if method == "probability":
        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue
                
            # normalize the weight
            ref_w = ref_state_dict[module_name + ".weight"]
            ref_w = ref_w/ref_w.std()
            # normalized_w = ref_w.clamp(-0.99, 0.99)    # restrict the weight to (-1,1)
            # abs_normalized_w = normalized_w.abs()

            idx = torch.logical_and(ref_w>-1, ref_w<1)
            prob_p1 = torch.where(ref_w <= -1, p_min/2, ref_w)
            prob_p1 = torch.where(ref_w > 1, p_max, prob_p1)
            prob_p1[idx] = p_min/2 + 0.5*(p_max - p_min/2)*(ref_w[idx] + 1) 
            
            module.weight.latent_param.data = torch.log(-1 + 1/(1 - prob_p1))
    
    elif method == "plain":
        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue
                
            # normalize the weight
            # ref_w = ref_state_dict[module_name + ".weight"]
            # normalized_w = ref_w.clamp(-0.99, 0.99)    # restrict the weight to (-1,1)
            # abs_normalized_w = normalized_w.abs()

            module.weight.latent_param.data = torch.zeros_like(module.weight.latent_param.data)

    else:
        pass

def init_weights(module, init_type='kaiming', gain=0.01):
    '''
    initialize network's weights
    init_type: normal | uniform | kaiming  
    '''
    classname = module.__class__.__name__
    if hasattr(module, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(module.weight.data, 0.0, gain)
        elif init_type == "uniform":
            nn.init.uniform_(module.weight.data, a=-1, b=1)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')

        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif (classname.find('BatchNorm') != -1 and module.weight is not None):
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)

    elif (classname.find("GroupNorm") != -1 and module.weight is not None):
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)