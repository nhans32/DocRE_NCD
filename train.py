import torch
import transformers
import const
from utils import batch_to_device, add_train_mask
from tqdm import tqdm
from eval_official import to_official, official_evaluate, detailed_supervised_evaluate
from torch.utils.data import DataLoader
from utils import collate_fn
import os
from eval_candidates import candidates_evaluate
from utils import create_dirs, empty_dir
import json
import torch.nn.functional as F
from eval_cluster import cluster_evaluate
from model import DocRedModel
import json

def load_optim_sched(model,
                     train_dataloader,
                     num_epochs,
                     encoder_lr):
    # -- OPTIMIZER --
    encoder_params = ['luke_model'] # Want to set lr on base model to 3e-5, and rest of model to 1e-4 with eps 1e-6
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in encoder_params)], 'lr': encoder_lr}, # NOTE: modify learning rate for this
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in encoder_params)]}
    ], lr=1e-4, eps=1e-6)

    # -- SCHEDULER --
    total_steps = int(len(train_dataloader) * num_epochs)
    warmup_steps = int(total_steps * const.WARMUP_RATIO)
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    return optimizer, scheduler



def train_epoch(model : DocRedModel,
                optimizer,
                scheduler,
                train_dataloader,
                cur_epoch,
                num_epochs,
                cur_steps):
    
    running_losses = {
        'sup_loss': 0.0,
        'tot_contr_loss': 0.0,
        'sup_contr_loss': 0.0,
        'unsup_contr_loss': 0.0
    }
    batch_i = 0

    model.zero_grad()
    model.train()
    with tqdm(train_dataloader, unit='batch') as tepoch:
        for batch in tepoch:
            tepoch.set_description(f'Train Epoch {cur_epoch+1}/{num_epochs}')

            batch = batch_to_device(batch, const.DEVICE)

            model.zero_grad()

            _, _, losses = model(batch=batch, train=True)

            losses['update_loss'].backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), const.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            cur_steps += 1
            batch_i += 1

            for l in running_losses.keys():
                running_losses[l] += losses[l].item()

            display_losses = {k: v/batch_i for k, v in running_losses.items()}
            tepoch.set_postfix(display_losses)

    return cur_steps, display_losses



def validate_epoch(model : DocRedModel,
                   dataloader):
    embeddings, predictions, labels, labels_original, train_mask = [], [], [], [], []

    model.eval()
    with tqdm(dataloader, unit='batch') as vepoch:
        for batch in vepoch:
            vepoch.set_description(f"Validation")
            batch = batch_to_device(batch, const.DEVICE)

            with torch.no_grad():
                embeds, preds, _ = model(batch=batch, train=False)

                preds = preds.cpu()
                embeds = embeds.cpu()

                preds[torch.isnan(preds)] = 0
                predictions.append(preds)
                embeddings.append(embeds)

                for i in range(len(batch['labels'])):
                    labels.append(torch.tensor(batch['labels'][i]))
                    labels_original.append(torch.tensor(batch['labels_original'][i]))
                    train_mask.append(batch['train_mask'][i])

    embeddings = torch.cat(embeddings, dim=0).to(torch.float32)
    predictions = torch.cat(predictions, dim=0).to(torch.float32)
    labels = torch.cat(labels, dim=0).to(torch.float32)
    labels_original = torch.cat(labels_original, dim=0).to(torch.float32)
    train_mask = torch.cat(train_mask, dim=0).to(torch.bool)

    return embeddings, predictions, labels, labels_original, train_mask



def train_official(model : DocRedModel,
                   train_dataloader,
                   dev_dataloader,
                   dev_samples,
                   id2rel_holdout,
                   encoder_lr,
                   num_epochs,
                   out_dir):
    
    if model.mode != const.MODE_OFFICIAL:
        raise ValueError("Model mode must be MODE_OFFICIAL")

    optimizer, scheduler = load_optim_sched(model, train_dataloader, num_epochs, encoder_lr)
    checkpoint_dir, stats_dir, plots_dir = create_dirs(out_dir)

    cur_steps = 0
    stats = []
    best_stat = -1

    for epoch in range(num_epochs):
        cur_steps, display_losses = train_epoch(model, optimizer, scheduler, train_dataloader, epoch, num_epochs, cur_steps)
        _, predictions, labels, _, _ = validate_epoch(model, dev_dataloader)

        # -- DocRED Official Evaluation --
        official_predictions = to_official(predictions, dev_samples, id2rel_holdout)
        if len(official_predictions) > 0:
            official_stats = official_evaluate(official_predictions, const.DATA_DIR)
            print(official_stats)

            if official_stats['f1_ign'] > best_stat: # If achieve higher F1 score, save model
                empty_dir(checkpoint_dir)
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'best-checkpoint_epoch-{epoch+1}.pt'))
                best_stat = official_stats['f1_ign']
        else:
            official_stats = {'none': None}
            print(official_stats)

        # -- Micro and Macro F1 Stats --
        detailed_stats = detailed_supervised_evaluate(predictions, labels, id2rel_holdout)

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'latest-checkpoint.pt')) # Save model each epoch

        stats.append({
            'official_stats': official_stats,
            'detailed_stats': detailed_stats,
            'losses': display_losses,
            'epoch': epoch+1
        })
        json.dump(stats, open(os.path.join(stats_dir, 'stats.json'), 'w'), indent=2)



def train_contr_candidates(model : DocRedModel,
                           train_dataloader : DataLoader,
                           val_train_dataloader : DataLoader, # This is just the training samples treated as validation to get their embeddings
                           id2rel_original,
                           normalize_embeds, # Should we normalize the embeddings of validation output?
                           encoder_lr,
                           num_epochs,
                           out_dir):
    
    if model.mode != const.MODE_CONTRASTIVE_CANDIDATES:
        raise ValueError("Model mode must be MODE_CONTRASTIVE_CANDIDATES")
    if isinstance(val_train_dataloader.sampler, torch.utils.data.sampler.RandomSampler):
        raise ValueError("val_train_dataloader must have a deterministic sampler")

    optimizer, scheduler = load_optim_sched(model, train_dataloader, num_epochs, encoder_lr)
    checkpoint_dir, stats_dir, plots_dir = create_dirs(out_dir)

    cur_steps = 0
    stats = []

    for epoch in range(num_epochs):
        cur_steps, display_losses = train_epoch(model, optimizer, scheduler, train_dataloader, epoch, num_epochs, cur_steps)

        embeddings, _, labels, labels_original, _ = validate_epoch(model, val_train_dataloader)
        if normalize_embeds:
            embeddings = F.normalize(embeddings, dim=-1)

        cand_stats, cand_mask = candidates_evaluate(embeddings, labels, labels_original, id2rel_original, os.path.join(plots_dir, f'candidate_epoch-{epoch+1}.png'))
        cand_stats['losses'] = display_losses
        cand_stats['epoch'] = epoch+1

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'latest-checkpoint.pt')) # Save model each epoch
        torch.save(cand_mask, os.path.join(checkpoint_dir, f'latest-contr-cand-mask.pt'))

        stats.append(cand_stats)
        json.dump(stats, open(os.path.join(stats_dir, 'stats.json'), 'w'), indent=2)



def train_contr_cluster(model : DocRedModel,
                        train_samples,
                        train_dataloader : DataLoader,
                        val_train_dataloader : DataLoader,
                        cand_mask, # This is the mask generated from contrastive_evaluate
                        id2rel_original,
                        id2rel_holdout,
                        normalize_embeds,
                        encoder_lr,
                        last_epoch,
                        num_epochs,
                        out_dir):
                
    if model.mode != const.MODE_CONTRASTIVE_CLUSTER:
        raise ValueError("Model mode must be MODE_CONTRASTIVE_CLUSTER")

    add_train_mask(train_samples, cand_mask)
        
    optimizer, scheduler = load_optim_sched(model, train_dataloader, num_epochs, encoder_lr)
    checkpoint_dir, stats_dir, plots_dir = create_dirs(out_dir)

    cur_steps = 0
    stats = []

    for epoch in range(last_epoch, num_epochs):
        cur_steps, display_losses = train_epoch(model, optimizer, scheduler, train_dataloader, epoch, num_epochs, cur_steps)

        embeddings, _, labels, labels_original, train_mask = validate_epoch(model, val_train_dataloader)
        if normalize_embeds:
            embeddings = F.normalize(embeddings, dim=-1)

        cluster_stats, pseudolabels, id2rel_holdout_update = cluster_evaluate(embeddings, train_mask, labels, labels_original, id2rel_original, id2rel_holdout, os.path.join(plots_dir, f'cluster_epoch-{epoch+1}.png'))
        cluster_stats['losses'] = display_losses
        cluster_stats['epoch'] = epoch+1

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'latest-checkpoint.pt'))
        torch.save(pseudolabels, os.path.join(checkpoint_dir, f'latest-pseudolabels.pt'))
        json.dump(id2rel_holdout_update, open(os.path.join(checkpoint_dir, 'latest-id2rel-holdout-update.json'), 'w'), indent=2)

        stats.append(cluster_stats)
        json.dump(stats, open(os.path.join(stats_dir, 'stats.json'), 'w'), indent=2)

    for doc in tqdm(train_samples, desc="Removing Train Mask..."):
        doc.pop('train_mask') # Remove train mask


