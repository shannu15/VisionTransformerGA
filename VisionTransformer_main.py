import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

import random
import tqdm
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.optim as optim
from TrainValidateWrapper import TrainValidateWrapper
from SimpleTransformer import SimpleTransformer
from PatchEmbedding import PatchEmbedding_CNN
import Utils
import sys
import math
import os

# ------constants------------
NUM_EPOCHS = 200
BATCH_SIZE = 16
GRADIENT_ACCUMULATE_EVERY = 1
LEARNING_RATE = 1e-4 # does not learn with 1e-3
VALIDATE_EVERY = 1
SEQ_LENGTH = 197 
RESUME_TRAINING = True # 14x14 + 1 for cls_token
# set to false to start training from beginning
#---------------------------
best_test_accuracy = 0

def count_parameters(model): # count number of trainable parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def configure_optimizers(mymodel):
    """
    This long function is unfortunately doing something very simple and is being
    very defensive:
    We are separating out all parameters of the model into two buckets: those that
    will experience
    weight decay for regularization and those that won't (biases, and
    layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    #,torch.nn.Parameter, torch.nn.Conv2d
    for mn, m in mymodel.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif fpn.startswith('model.token_emb'):
                no_decay.add(fpn) # change to no_decay if using PatchEmbedding_CNN
    param_dict = {pn: p for pn, p in mymodel.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
    % (str(param_dict.keys() - union_params), )
    
    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.1},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9, 0.95))
    return optimizer

def main():
    global best_test_accuracy
    vision_model = SimpleTransformer(
        dim = 768, # embedding
        num_unique_tokens = 10, # for CIFAR-10, use 100 for CIFAR-100
        num_layers = 12,
        heads = 8,
        max_seq_len = SEQ_LENGTH,
    ).to(device)
    model = TrainValidateWrapper(vision_model)
    model.to(device)
    pcount = count_parameters(model)
    print("count of parameters in the model = ", pcount/1e6, " million")
    train_loader, val_loader, testset = Utils.get_loaders_cifar(dataset_type="CIFAR10", img_width=224, img_height=224,
    batch_size=BATCH_SIZE)
    
    optim = configure_optimizers(model)
    
    # --------training---------
    checkpoint_dir = "checkpoint"  # Define checkpoint directory
    ckpt_path = os.path.join(checkpoint_dir, "visiontrans_model.pt")

    # Ensure checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if RESUME_TRAINING == False:
        start = 0
    else:
        if os.path.exists(ckpt_path):
            checkpoint_data = torch.load(ckpt_path)
            model.load_state_dict(checkpoint_data['state_dict'])
            optim.load_state_dict(checkpoint_data['optimizer'])
            start = checkpoint_data['epoch']
            best_test_accuracy = checkpoint_data['test_acc']
            print('best test accuracy from restored model=', best_test_accuracy)
        else:
            print(f"No checkpoint found at {ckpt_path}. Starting training from scratch.")
            start = 0

    for i in tqdm.tqdm(range(start, NUM_EPOCHS), mininterval = 10., desc = 'training'):
        for k, data in enumerate(train_loader):
            model.train()
            total_loss = 0
            for __ in range(GRADIENT_ACCUMULATE_EVERY):
                x, y = data #next(train_loader)
                x = x.to(device)
                y = y.to(device)
                loss = model(x, y)
                loss.backward()
            if (k % 500 == 0):
                print(f'training loss: {loss.item()} -- iteration = {k}')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad()
        
        if i % VALIDATE_EVERY == 0:
            model.eval()
            val_count = 3000
            total_count = 0
            count_correct = 0
            with torch.no_grad():
                for v, data in enumerate(val_loader):
                    x, y = data
                    x = x.to(device)
                    y = y.to(device)
                    count_correct = count_correct + model.validate(x, y)
                    total_count = total_count + x.shape[0]
                accuracy = (count_correct / total_count) * 100
                print("\n-------------Test Accuracy = ", accuracy, "\n")
            
            if accuracy > best_test_accuracy:
                print("----------saving model-----------------")
                checkpoint_data = {
                    'epoch': i,
                    'state_dict': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'test_acc': accuracy
                }
                torch.save(checkpoint_data, ckpt_path)
                best_test_accuracy = accuracy
            model.train()

if __name__ == "__main__":
    sys.exit(int(main() or 0))
