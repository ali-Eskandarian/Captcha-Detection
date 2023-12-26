from tqdm import tqdm
import torch
import config

def train_fn(model, data_loader, optimizer):
    model.train()
    total_training_loss = 0
    for batch_data in tqdm(data_loader, total=len(data_loader), desc="Training"):
        for key, value in batch_data.items():
            batch_data[key] = value.to(config.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**batch_data)
        loss.backward()
        optimizer.step()
        total_training_loss += loss.item()
    return total_training_loss / len(data_loader)

def eval_fn(model, data_loader):
    model.eval()
    total_evaluation_loss = 0
    all_predictions = []
    for batch_data in tqdm(data_loader, total=len(data_loader), desc="Evaluating"):
        for key, value in batch_data.items():
            batch_data[key] = value.to(config.DEVICE)
        batch_predictions, loss = model(**batch_data)
        total_evaluation_loss += loss.item()
        all_predictions.append(batch_predictions)
    return all_predictions, total_evaluation_loss / len(data_loader)
