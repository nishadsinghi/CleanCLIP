import time
import wandb
import torch
import logging
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast

def get_loss(umodel, outputs, criterion, options, gather_backdoor_indices):  
    if(options.inmodal):
        image_embeds, augmented_image_embeds = outputs.image_embeds[:len(outputs.image_embeds) // 2], outputs.image_embeds[len(outputs.image_embeds) // 2:]
        text_embeds, augmented_text_embeds = outputs.text_embeds[:len(outputs.text_embeds) // 2], outputs.text_embeds[len(outputs.text_embeds) // 2:]
    else:
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
            
    if(options.distributed):
        if(options.inmodal):
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]
            augmented_gathered_image_embeds = [torch.zeros_like(augmented_image_embeds) for _ in range(options.num_devices)]
            augmented_gathered_text_embeds = [torch.zeros_like(augmented_text_embeds) for _ in range(options.num_devices)]
            
            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)
            dist.all_gather(augmented_gathered_image_embeds, augmented_image_embeds)
            dist.all_gather(augmented_gathered_text_embeds, augmented_text_embeds)
            
            image_embeds = torch.cat(gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
            text_embeds  = torch.cat(gathered_text_embeds[:options.rank]+ [text_embeds] + gathered_text_embeds[options.rank + 1:])
            augmented_image_embeds = torch.cat(augmented_gathered_image_embeds[:options.rank] + [augmented_image_embeds] + augmented_gathered_image_embeds[options.rank + 1:])
            augmented_text_embeds  = torch.cat(augmented_gathered_text_embeds[:options.rank]+ [augmented_text_embeds] + augmented_gathered_text_embeds[options.rank + 1:])      
        else:
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]

            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)

            image_embeds = torch.cat(gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
            text_embeds  = torch.cat(gathered_text_embeds[:options.rank]+ [text_embeds] + gathered_text_embeds[options.rank + 1:])

    constraint = torch.tensor(0).to(options.device)
    if options.unlearn:
        normal_indices = (~gather_backdoor_indices).nonzero().squeeze()
        backdoor_indices = gather_backdoor_indices.nonzero()
        backdoor_indices = backdoor_indices[:,0] if len(backdoor_indices.shape) == 2 else backdoor_indices
        if len(backdoor_indices):
            backdoor_image_embeds = image_embeds[backdoor_indices]
            backdoor_text_embeds  = text_embeds[backdoor_indices]
            similarity_backdoor_embeds = torch.diagonal(backdoor_image_embeds @ backdoor_text_embeds.t())
            constraint = (similarity_backdoor_embeds + options.unlearn_target).square().mean().to(options.device, non_blocking = True)
        image_embeds = image_embeds[normal_indices]
        text_embeds  = text_embeds[normal_indices]
        
    logits_text_per_image = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
    logits_image_per_text = logits_text_per_image.t()

    if(options.inmodal):
        logits_image_per_augmented_image = umodel.logit_scale.exp() * image_embeds @ augmented_image_embeds.t()
        logits_text_per_augmented_text = umodel.logit_scale.exp() * text_embeds @ augmented_text_embeds.t()

    batch_size = len(logits_text_per_image)
    target = torch.arange(batch_size).long().to(options.device, non_blocking = True)
    
    contrastive_loss = torch.tensor(0).to(options.device)
    if(options.inmodal):
        crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2
        inmodal_contrastive_loss = (criterion(logits_image_per_augmented_image, target) + criterion(logits_text_per_augmented_text, target)) / 2
        # contrastive_loss = (crossmodal_contrastive_loss + inmodal_contrastive_loss) / 2
        contrastive_loss = (options.clip_weight * crossmodal_contrastive_loss) + (options.inmodal_weight * inmodal_contrastive_loss)
    else:
        crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2
        contrastive_loss = crossmodal_contrastive_loss

    if options.unlearn:
        contrastive_loss = contrastive_loss + (options.constraint_weight * constraint)

    loss = contrastive_loss
    return loss, contrastive_loss, constraint

# @torch.no_grad()
# def get_clean_batch(model, batch, options, step, threshold = 0.6):
#     input_ids, attention_mask, pixel_values, pixel_values_cropped = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True), batch["pixel_values_cropped"].to(options.device, non_blocking = True)
#     pixel_values_all = torch.cat([pixel_values, pixel_values_cropped])
#     outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values_all)
#     image_embeds = outputs.image_embeds
#     image_embeds, image_embeds_cropped = image_embeds[: len(image_embeds) // 2], image_embeds[len(image_embeds) // 2 :] 
#     pairwise_similarity = 1 - (((image_embeds - image_embeds_cropped)**2).sum(dim = 1) / 2)
#     is_normal = pairwise_similarity > threshold ## if the pairwise similarity is high the it is an original image 
#     indices = is_normal.nonzero().squeeze()
#     # indices = range(len(pixel_values)) if len(indices) == 0 else indices ## don't want any empty batch

#     is_backdoor = batch["is_backdoor"].to(options.device, non_blocking = True)
#     total_backdoors = sum(is_backdoor).item()
#     predicted_backdoor = ~ is_normal  
#     fraction_caught = -1

#     if sum(predicted_backdoor).item() != len(predicted_backdoor): 
#         backdoor_predicted_equal = is_backdoor & predicted_backdoor
#         correct_backdoors = sum(backdoor_predicted_equal).item()
#         if total_backdoors > 0:
#             fraction_caught = correct_backdoors // total_backdoors

#     if options.wandb and options.master:
#         wandb.log({f'{options.rank}/len of indices' : len(indices), 'step': step})
#         wandb.log({f'{options.rank}/# images removed' : len(pixel_values) - len(indices), 'step': step})
#         wandb.log({f'{options.rank}/total backdoors' : total_backdoors, 'step': step})      
#         wandb.log({f'{options.rank}/correct backdoors detected' : correct_backdoors, 'step': step})      
#         wandb.log({f'{options.rank}/fraction of backdoors caught' : fraction_caught, 'step': step})      

    # return input_ids[indices], attention_mask[indices], pixel_values[indices], torch.tensor(len(indices)).to(options.device) 
    # return is_normal

def process_batch(model, batch, options, step):
    input_ids, attention_mask, pixel_values, is_backdoor = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True), batch["is_backdoor"].to(options.device, non_blocking = True)
    outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
    with torch.no_grad():
        similarity = torch.diagonal(outputs.image_embeds @ outputs.text_embeds.t())
        topmax     = int(options.remove_fraction * len(similarity))
        detect_indices = similarity.topk(topmax).indices
    num_backdoor = is_backdoor.sum().item()
    backdoor_indices = is_backdoor.nonzero()
    backdoor_indices = backdoor_indices[:,0] if len(backdoor_indices.shape) == 2 else backdoor_indices
    count = 0
    if len(backdoor_indices) > 0:
        for backdoor_index in backdoor_indices:
            count += (backdoor_index in detect_indices)
    if options.wandb and options.master:
        wandb.log({f'{options.rank}/total backdoors' : num_backdoor, 'step': step})      
        wandb.log({f'{options.rank}/correct backdoors detected' : count, 'step': step})   
    pred_backdoor_indices = torch.zeros_like(similarity).int()
    pred_backdoor_indices[detect_indices] = 1
    return outputs, pred_backdoor_indices

def train(epoch, model, data, optimizer, scheduler, scaler, options):    
    dataloader = data["train"]
    if(options.distributed): dataloader.sampler.set_epoch(epoch)

    model.train()
    criterion = nn.CrossEntropyLoss().to(options.device) #if not options.unlearn else nn.CrossEntropyLoss(reduction = 'none').to(options.device)

    modulo = max(1, int(dataloader.num_samples / options.batch_size / 5))
    umodel = model.module if(options.distributed) else model

    start = time.time()
    
    logging.info(f"Num samples: {dataloader.num_samples}, Num_batches: {dataloader.num_batches}")
    for index, batch in enumerate(dataloader): 
        step = dataloader.num_batches * epoch + index
        scheduler(step)

        optimizer.zero_grad()
        
        if(options.inmodal):
            input_ids, attention_mask, pixel_values = batch["input_ids"][0].to(options.device, non_blocking = True), batch["attention_mask"][0].to(options.device, non_blocking = True), batch["pixel_values"][0].to(options.device, non_blocking = True)
            augmented_input_ids, augmented_attention_mask, augmented_pixel_values = batch["input_ids"][1].to(options.device, non_blocking = True), batch["attention_mask"][1].to(options.device, non_blocking = True), batch["pixel_values"][1].to(options.device, non_blocking = True)
            input_ids = torch.cat([input_ids, augmented_input_ids])
            attention_mask = torch.cat([attention_mask, augmented_attention_mask])
            pixel_values = torch.cat([pixel_values, augmented_pixel_values])
        else:
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True)

        gather_backdoor_indices = None
        if options.unlearn:
            if options.distributed:
                backdoor_indices = batch["is_backdoor"].to(options.device)
                gather_backdoor_indices = [torch.zeros_like(backdoor_indices) for _ in range(options.num_devices)]
                dist.all_gather(tensor_list = gather_backdoor_indices, tensor = backdoor_indices)
                gather_backdoor_indices = torch.cat(gather_backdoor_indices).to(options.device, non_blocking = True)
            else:
                gather_backdoor_indices = batch["is_backdoor"].to(options.device, non_blocking = True)
        outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)

        with autocast():
            loss, contrastive_loss, constraint_loss = get_loss(umodel, outputs, criterion, options, gather_backdoor_indices)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        
        scaler.update()
        umodel.logit_scale.data = torch.clamp(umodel.logit_scale.data, 0, 4.6052)

        end = time.time()

        if(options.master and (((index + 1) % modulo == 0) or (index == dataloader.num_batches - 1))):
            num_samples = (index + 1) * len(input_ids) * options.num_devices
            dataloader_num_samples = dataloader.num_samples

            logging.info(f"Train Epoch: {epoch:02d} [{num_samples}/{dataloader_num_samples} ({100.0 * (index + 1) / dataloader.num_batches:.0f}%)]\tLoss: {loss.item():.6f}\tTime taken {end - start:.3f}\tLearning Rate: {optimizer.param_groups[0]['lr']:.9f}")

            metrics = {"loss": loss.item(), "contrastive_loss": contrastive_loss.item(), "constraint_loss": constraint_loss.item(), "time": end - start, "lr": optimizer.param_groups[0]["lr"]}
            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"train/{key}": value, "step": step})
        
            start = time.time()
