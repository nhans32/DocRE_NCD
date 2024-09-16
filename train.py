import torch
import transformers
import const
from utils import batch_to_device
from tqdm import tqdm
from evaluation import to_official, official_evaluate

def load_optim_sched(model,
                     train_dataloader,
                     num_epochs,
                     warmup_ratio=const.WARMUP_RATIO):
    # -- OPTIMIZER --
    encoder_params = ['luke_model'] # Want to set lr on base model to 3e-5, and rest of model to 1e-4 with eps 1e-6
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in encoder_params)], 'lr': 3e-5},
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in encoder_params)]}
    ], lr=1e-4, eps=1e-6)

    # -- SCHEDULER --
    total_steps = int(len(train_dataloader) * num_epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    return optimizer, scheduler



def train_epoch(model,
                optimizer,
                scheduler,
                train_dataloader,
                cur_epoch,
                num_epochs,
                cur_steps):
    running_loss = 0.0
    batch_i = 0

    model.zero_grad()
    model.train()
    with tqdm(train_dataloader, unit='batch') as tepoch:
        for batch in tepoch:
            tepoch.set_description(f'Train Epoch {cur_epoch + 1}/{num_epochs}')
            batch = batch_to_device(batch, const.DEVICE)

            model.zero_grad()

            _, _, loss = model(batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), const.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            cur_steps += 1
            batch_i += 1

            tepoch.set_postfix(loss=running_loss / batch_i)
    return cur_steps



def validate(model,
             dataloader):
    embeddings, predictions, labels = [], [], []

    model.eval()
    with tqdm(dataloader, unit='batch') as vepoch:
        for batch in vepoch:
            vepoch.set_description(f"Validation")
            batch = batch_to_device(batch, const.DEVICE)

            with torch.no_grad():
                embeds, preds, _ = model(batch)

                preds = preds.cpu()
                embeds = embeds.cpu()

                preds[torch.isnan(preds)] = 0
                predictions.append(preds)
                embeddings.append(embeds)

                for i in range(len(batch['labels'])):
                    labels.append(batch['labels'][i])

    embeddings = torch.cat(embeddings, dim=0).to(torch.float32)
    predictions = torch.cat(predictions, dim=0).to(torch.float32)

    return embeddings, predictions, labels



def train(model,
          train_dataloader,
          dev_dataloader,
          dev_samples,
          id2rel,
          num_epochs):
    
    optimizer, scheduler = load_optim_sched(model, train_dataloader, num_epochs)

    cur_steps = 0
    for epoch in range(num_epochs):
        cur_steps = train_epoch(model, optimizer, scheduler, train_dataloader, epoch, num_epochs, cur_steps)
        embeddings, predictions, labels = validate(model, dev_dataloader)

        official_preds = to_official(preds=predictions,
                                     samples=dev_samples,
                                     id2rel=id2rel)
        
        if len(official_preds) > 0:
            f1, _, f1_ign, _ = official_evaluate(official_preds, const.DATA_DIR)
            print(f"Epoch {epoch + 1}/{num_epochs} F1: {f1:.4f} F1 Ign: {f1_ign:.4f}")
        else:
            print(f"No predictions made for epoch {epoch+1}...")
            continue