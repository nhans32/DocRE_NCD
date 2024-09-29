import torch
import transformers
import const
from utils import batch_to_device
from tqdm import tqdm
from eval_official import to_official, official_evaluate, detailed_supervised_evaluate
from torch.utils.data import DataLoader
from utils import collate_fn
import os
from eval_contrastive import contrastive_evaluate
from utils import print_dict
import json

def load_optim_sched(model,
                     train_dataloader,
                     num_epochs,
                     encoder_lr=3e-5,
                     other_lr=1e-4):
    # -- OPTIMIZER --
    encoder_params = ['luke_model'] # Want to set lr on base model to 3e-5, and rest of model to 1e-4 with eps 1e-6
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in encoder_params)], 'lr': encoder_lr}, # NOTE: modify learning rate for this
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in encoder_params)]}
    ], lr=other_lr, eps=1e-6)

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
                cur_steps):
    
    running_losses = {
        'sup_loss': 0.0,
        'contr_loss': 0.0,
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



def validate_epoch(model,
                   dataloader):
    embeddings, predictions, labels, labels_original = [], [], [], []

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

    embeddings = torch.cat(embeddings, dim=0).to(torch.float32)
    predictions = torch.cat(predictions, dim=0).to(torch.float32)
    labels = torch.cat(labels, dim=0).to(torch.float32)
    labels_original = torch.cat(labels_original, dim=0).to(torch.float32)

    return embeddings, predictions, labels, labels_original



def train(model,
          train_dataloader,
          dev_dataloader,
          val_train_dataloader, # dataloader with training samples but not shuffled
          dev_samples,
          id2rel_holdout,
          id2rel_original,
          num_epochs,
          encoder_lr,
          out_dir):
    
    optimizer, scheduler = load_optim_sched(model, 
                                            train_dataloader, 
                                            num_epochs,
                                            encoder_lr=encoder_lr)

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
        if model.mode == const.MODE_OFFICIAL:
            cur_steps, display_losses = train_epoch(model, optimizer, scheduler, train_dataloader, epoch, num_epochs, cur_steps)

            _, predictions, labels, labels_original = validate_epoch(model, dev_dataloader)

            detailed_sup_stats = detailed_supervised_evaluate(predictions=predictions,
                                                              labels=labels,
                                                              labels_original=labels_original)

            official_predictions = to_official(preds=predictions, # This is the original DocRED
                                               samples=dev_samples,
                                               id2rel=id2rel_holdout)
            if len(official_predictions) > 0:
                official_stats = official_evaluate(official_predictions, const.DATA_DIR)

                if official_stats['f1_ign'] > best_stat: # If achieve higher F1 score, save model
                    for f in os.listdir(checkpoint_dir): # empty the checkpoint directory
                        os.remove(os.path.join(checkpoint_dir, f))
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'best-checkpoint_epoch-{epoch}.pt'))
                    best_stat = official_stats['f1_ign']
            else:
                print(f"No predictions made...")
                official_stats = {'none': None}
            
            official_stats['losses'] = display_losses
            stats.append(official_stats)

        elif model.mode == const.MODE_CONTRASTIVE: # We don't care about performance on dev set for contrastive learning, we just want the embeddings for the training set
            # TODO: add initial validation step
            cur_steps, display_losses = train_epoch(model, optimizer, scheduler, train_dataloader, epoch, num_epochs, cur_steps)

            embeddings, _, labels, labels_original = validate_epoch(model, val_train_dataloader)
            contr_stats = contrastive_evaluate(embeddings=embeddings, 
                                               labels=labels,
                                               labels_original=labels_original,
                                               id2rel_original=id2rel_original,
                                               plot_save_path=os.path.join(plots_dir, f'contr_epoch-{epoch}.png'))
            contr_stats['losses'] = display_losses
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
