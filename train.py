import torch
import transformers
import const
from utils import batch_to_device
from tqdm import tqdm
from eval_official import to_official, official_evaluate
from torch.utils.data import DataLoader
from utils import collate_fn
import os
from eval_contrastive import contrastive_evaluate
from utils import print_dict
import json

def load_optim_sched(model,
                     train_dataloader,
                     num_epochs,
                     encoder_lr=3e-5):
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



def train_epoch(model,
                optimizer,
                scheduler,
                train_dataloader,
                cur_epoch,
                num_epochs,
                cur_steps,
                mode):
    running_sup_loss, running_contr_loss = 0.0, 0.0
    batch_i = 0

    max_num_pairs = 0
    num_batch_skipped = 0

    model.zero_grad()
    model.train()
    with tqdm(train_dataloader, unit='batch') as tepoch:
        for batch in tepoch:
            num_pairs = sum([len(p) for p in batch['hts']]) # Number of pairs in batch
            if mode == const.MODE_CONTRASTIVE and num_pairs > const.MAX_BATCH_PAIRS:
                num_batch_skipped += 1
                continue

            tepoch.set_description(f'Train Epoch {cur_epoch+1}/{num_epochs} | Num. Pair Batch: {num_pairs}')

            batch = batch_to_device(batch, const.DEVICE)

            model.zero_grad()

            _, _, losses = model(batch=batch, mode=mode, train=True)

            losses['update_loss'].backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), const.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            running_sup_loss += losses['sup_loss'].item()
            running_contr_loss += losses['contr_loss'].item()

            cur_steps += 1
            batch_i += 1

            if num_pairs > max_num_pairs:
                max_num_pairs = num_pairs
            tepoch.set_postfix(num_batch_skipped=num_batch_skipped, max_num_pairs=max_num_pairs, sup_loss=running_sup_loss / batch_i, contr_loss=running_contr_loss / batch_i)

    return cur_steps



def validate_epoch(model,
                   dataloader,
                   mode):
    embeddings, predictions, labels, labels_original = [], [], [], []

    model.eval()
    with tqdm(dataloader, unit='batch') as vepoch:
        for batch in vepoch:
            vepoch.set_description(f"Validation")
            batch = batch_to_device(batch, const.DEVICE)

            with torch.no_grad():
                embeds, preds, _ = model(batch=batch, mode=mode, train=False)

                preds = preds.cpu()
                embeds = embeds.cpu()

                preds[torch.isnan(preds)] = 0
                predictions.append(preds)
                embeddings.append(embeds)

                for i in range(len(batch['labels'])):
                    labels.append(torch.tensor(batch['labels'][i]))
                    labels_original.append(torch.tensor(batch['labels_original'][i]))

    embeddings = torch.cat(embeddings, dim=0).to(torch.float32)
    predictions = torch.cat(predictions, dim=0).to(torch.float32)
    labels = torch.cat(labels, dim=0).to(torch.float32)
    labels_original = torch.cat(labels_original, dim=0).to(torch.float32)

    return embeddings, predictions, labels, labels_original



def train(model,
          train_dataloader,
          dev_dataloader,
          train_samples,
          dev_samples,
          id2rel_holdout,
          id2rel_original,
          num_epochs,
          mode,
          out_dir):
    
    optimizer, scheduler = load_optim_sched(model, train_dataloader, num_epochs)

    val_train_dataloader = DataLoader(train_samples, 
                                      batch_size=dev_dataloader.batch_size,
                                      shuffle=isinstance(dev_dataloader.sampler, torch.utils.data.sampler.RandomSampler), # If sampler is RandomSampler, shuffle=True, else False
                                      collate_fn=collate_fn, 
                                      drop_last=dev_dataloader.drop_last) # (Only used in contrastive) Need new dataloader to validate over training data, i.e. drop_last=False. Want the Dataloader attributes of dev_dataloader just applied to train_samples

    # Create directory stuctures if it doesn't already exist
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    stats_dir = os.path.join(out_dir, 'stats')
    plots_dir = os.path.join(out_dir, 'plots')
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

    cur_steps = 0
    stats = []
    best_stat = -1
    for epoch in range(num_epochs):
        # -- SUPERVISED TRAINING --
        if mode == const.MODE_SUPERVISED:
            cur_steps = train_epoch(model, optimizer, scheduler, train_dataloader, epoch, num_epochs, cur_steps, mode)

            _, predictions, _, _ = validate_epoch(model, dev_dataloader, mode)
            official_predictions = to_official(preds=predictions,
                                               samples=dev_samples,
                                               id2rel=id2rel_holdout)
            if len(official_predictions) > 0:
                official_stats = official_evaluate(official_predictions, const.DATA_DIR)
                stats.append(official_stats)

                if official_stats['f1_ign'] > best_stat: # If achieve higher F1 score, save model
                    for f in os.listdir(checkpoint_dir): # empty the checkpoint directory
                        os.remove(os.path.join(checkpoint_dir, f))
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'best-checkpoint_epoch-{epoch}.pt'))
                    best_stat = official_stats['f1_ign']
            else:
                print(f"No predictions made...")
                stats.append(None)

        # -- CONTRASTIVE TRAINING --
        elif mode == const.MODE_CONTRASTIVE: # We don't care about performance on dev set for contrastive learning, we just want the embeddings for the training set
            if epoch == 0: # Just to see how the model performs on the training set before training
                embeddings, _, labels, labels_original = validate_epoch(model, val_train_dataloader, mode)
                contr_stats = contrastive_evaluate(embeddings=embeddings, 
                                                   labels=labels,
                                                   labels_original=labels_original,
                                                   id2rel_original=id2rel_original,
                                                   plot_save_path=os.path.join(plots_dir, f'contr_epoch-pre.png'))
                stats.append(contr_stats)
                stats[-1]['epoch'] = -1
            
            cur_steps = train_epoch(model, optimizer, scheduler, train_dataloader, epoch, num_epochs, cur_steps, mode)

            embeddings, _, labels, labels_original = validate_epoch(model, val_train_dataloader, mode)
            contr_stats = contrastive_evaluate(embeddings=embeddings, 
                                               labels=labels,
                                               labels_original=labels_original,
                                               id2rel_original=id2rel_original,
                                               plot_save_path=os.path.join(plots_dir, f'contr_epoch-{epoch}.png'))
            stats.append(contr_stats)

            if contr_stats['holdout_cand_ratio'] > best_stat: # If achieve higher holdout_cand_ratio, save model
                for f in os.listdir(checkpoint_dir): # empty the checkpoint directory
                    os.remove(os.path.join(checkpoint_dir, f))
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'best-checkpoint_epoch-{epoch}.pt'))
                best_stat = contr_stats['holdout_cand_ratio']

        stats[-1]['epoch'] = epoch

        with open(os.path.join(stats_dir, 'stats.json'), 'w') as f: # This will overwrite the file each epoch
            json.dump(stats, f, indent=2)

        print('--------')
