import os
os.environ["WANDB_API_KEY"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import sys
import time
import wandb
import torch
import logging
import warnings
import numpy as np
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from pkgs.openai.clip import load as load_model

from .train import train
from .evaluate import evaluate, Finetune
from .data import load as load_data
from .data import get_clean_train_dataloader, calculate_scores
from .parser import parse_args
from .scheduler import cosine_scheduler
from .logger import get_logger, set_logger

mp.set_start_method("spawn", force = True)
warnings.filterwarnings("ignore")


def gathered_elements_to_list(gather_elements):
    output = []
    for element in gather_elements:
        output = output + list(element)
    return output

def progressive_removal(options, model, processor, data, epoch):

    path = calculate_scores(options, model, data["train"], epoch)
    gather_path = [None for _ in range(options.num_devices)]
    if options.distributed:
        dist.all_gather_object(gather_path, path)
    
    if not options.master and options.distributed:
        logging.info(f'Device inside barrier 1 {options.device}')
        torch.distributed.barrier()
        logging.info(f'Device outside barrier 1 {options.device}')

    data["train"] = get_clean_train_dataloader(options, processor, path)

    options.train_data = path

    if options.master and options.distributed:
        logging.info(f'Device inside barrier 2 {options.device}')
        torch.distributed.barrier()
        logging.info(f'Device outside barrier 2 {options.device}')

    return options, data

def worker(rank, options, logger):
    options.rank = rank
    options.master = rank == 0
    
    set_logger(rank = rank, logger = logger, distributed = options.distributed)

    if(options.device == "cuda"):
        options.device += ":" + str(options.device_ids[options.rank] if options.distributed else options.device_id)

    logging.info(f"Using {options.device} device")

    if(options.master):
        logging.info("Params:")
        with open(os.path.join(options.log_dir_path, "params.txt"), "w") as file:
            for key in sorted(vars(options)):
                value = getattr(options, key)
                logging.info(f"{key}: {value}")
                file.write(f"{key}: {value}\n")

    if(options.distributed):
        dist.init_process_group(backend = options.distributed_backend, init_method = options.distributed_init_method, world_size = options.num_devices, rank = options.rank)
    
    options.batch_size = options.batch_size // options.num_devices

    model, processor = load_model(name = options.model_name, pretrained = options.pretrained)

    if(options.device == "cpu"):
        model.float()
    else:
        torch.cuda.set_device(options.device_ids[options.rank] if options.distributed else options.device_id)
        model.to(options.device)
        if(options.distributed):
            model = DDP(model, device_ids = [options.device_ids[options.rank]])
        
    data = load_data(options, processor)

    optimizer = None
    scheduler = None
    if(data["train"] is not None):        
        weight_decay_parameters = []
        no_weight_decay_parameters = []

        for name, parameter in model.named_parameters():
            if(all(key not in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                weight_decay_parameters.append(parameter)
                
            if(any(key in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                no_weight_decay_parameters.append(parameter)

        optimizer = optim.AdamW([{"params": no_weight_decay_parameters, "weight_decay": 0}, {"params": weight_decay_parameters, "weight_decay": options.weight_decay}], lr = options.lr, betas = (options.beta1, options.beta2), eps = options.eps)
        scheduler = cosine_scheduler(optimizer, options.lr, options.num_warmup_steps, data["train"].num_batches * options.epochs)

    start_epoch = 0
    if(options.checkpoint is not None):
        if(os.path.isfile(options.checkpoint)):
            checkpoint  = torch.load(options.checkpoint, map_location = options.device)
            start_epoch = 0 if options.complete_finetune else checkpoint['epoch'] 
            state_dict  = checkpoint["state_dict"]
            if(not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            # hack to load a non-distributed checkpoint for distributed training
            if (options.distributed and not next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {"module."+key: value for key, value in state_dict.items()}
            if(options.checkpoint_finetune):
                finetuned_checkpoint = torch.load(options.checkpoint_finetune, map_location = options.device)
                finetuned_state_dict = finetuned_checkpoint["state_dict"]
                for key in state_dict:
                    if 'visual' in key:
                        ft_key = name.replace("module.", "model.") if "module" in key else f'model.{key}'
                        state_dict[key] = finetuned_state_dict[ft_key]
                print('Loaded Visual Backbone from Finetuned Model')
            model.load_state_dict(state_dict)
            if(optimizer is not None): optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"Loaded checkpoint '{options.checkpoint}' (start epoch {checkpoint['epoch']})")
        else:
            logging.info(f"No checkpoint found at {options.checkpoint}")

    cudnn.benchmark = True
    cudnn.deterministic = False

    if(options.wandb and options.master):
        logging.debug("Starting wandb")
        wandb.init(project = "clip-defense", notes = options.notes, tags = [], config = vars(options), entity = 'mint-adobe')
        wandb.run.name = options.name
        wandb.save(os.path.join(options.log_dir_path, "params.txt"))

    evaluate(start_epoch, model, processor, data, options)

    if(data["train"] is not None):
        options.checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
        os.makedirs(options.checkpoints_dir_path, exist_ok = True)

        scaler = GradScaler()

        best_loss = np.inf

        if(options.progressive):
            options.progressive_epochs = list(map(int, options.progressive_epochs))
            if (start_epoch in options.progressive_epochs):
                options, data = progressive_removal(options, model, processor, data, start_epoch)

        for epoch in range(start_epoch + 1, options.epochs + 1):
            if(options.master): 
                logging.info(f"Starting Epoch {epoch}")

            start = time.time()
            train(epoch, model, data, optimizer, scheduler, scaler, options)
            end = time.time()

            if(options.master): 
                logging.info(f"Finished Epoch {epoch}, Time Taken: {end - start:.3f}")

            metrics = evaluate(epoch, model, processor, data, options)

            if(options.master):
                checkpoint = {"epoch": epoch, "name": options.name, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                if(options.complete_finetune):
                    torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch.pt"))
                else:
                    torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch_{epoch}.pt"))
                if("loss" in metrics):
                    if(metrics["loss"] < best_loss):
                        best_loss = metrics["loss"]
                        torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch.best.pt"))
            
            if(options.progressive):
                if epoch in options.progressive_epochs:
                    options, data = progressive_removal(options, model, processor, data, epoch)
            
                if epoch == options.stop_epoch:
                    return

    if(options.distributed):
        dist.destroy_process_group()

    if(options.wandb and options.master):
        wandb.finish()

if(__name__ == "__main__"):    
    options = parse_args()

    options.log_dir_path = os.path.join(options.logs, options.name)
    options.log_file_path = os.path.join(options.log_dir_path, "output.log")
    
    os.makedirs(options.log_dir_path, exist_ok = True)
    logger, listener = get_logger(options.log_file_path)

    listener.start()

    ngpus = torch.cuda.device_count()
    if(ngpus == 0 or options.device == "cpu"):
        options.device = "cpu"
        options.num_devices = 1
        options.distributed = False
        worker(0, options, logger)
    else:
        if(ngpus == 1 or not options.distributed):
            options.device = "cuda"
            options.num_devices = 1
            options.distributed = False
            worker(0, options, logger)
        else:
            options.device = "cuda"
            if(options.device_ids is None):
                options.device_ids = list(range(ngpus))
                options.num_devices = ngpus
            else:
                options.device_ids = list(map(int, options.device_ids))
                options.num_devices = len(options.device_ids)
            options.distributed = True
            os.environ["NCCL_P2P_DISABLE"] = "1"
            mp.spawn(worker, nprocs = options.num_devices, args = (options, logger))
    
    listener.stop()
