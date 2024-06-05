import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 
from tqdm import tqdm
import numpy as np

# epoch - Train
def Train_multi(dataloader, device, model, loss_fn, optimizer):
    loss_list, acc_list, micro_f1_list, macro_f1_list = [], [], [], []

    for batch_i, batch_loader in enumerate(tqdm(dataloader)):
        batch_l, batch_b, batch_r, batch_y = batch_loader
        batch_l, batch_b, batch_r, batch_y = batch_l.to(device), batch_b.to(device), batch_r.to(device), batch_y.to(device)

        model.train()
        pred = model(batch_l.float(), batch_b.float(), batch_r.float())
        loss = loss_fn(pred, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_y = torch.max(pred, 1).indices
        # print(pred_y)
        acc = accuracy_score(batch_y.cpu(), pred_y.cpu())
        micro_f1 = f1_score(batch_y.cpu(), pred_y.cpu(), average='micro')
        macro_f1 = f1_score(batch_y.cpu(), pred_y.cpu(), average='macro')
        loss = loss.cpu()
        
        loss_list.append(loss.item())
        acc_list.append(acc)
        micro_f1_list.append(micro_f1)
        macro_f1_list.append(macro_f1)
    print(f'loss: {np.mean(loss_list):.4f}')
    print(f'acc: {np.mean(acc_list):.4f}')
    print(f'micro_f1: {np.mean(micro_f1_list):.4f}')
    print(f'macro_f1: {np.mean(macro_f1_list):.4f}')

def Test_multi(dataloader, device, model, loss_fn):
    avg_loss = 0
    total_pred, total_y = [], []

    for batch_i, batch_loader in enumerate(tqdm(dataloader)):
        batch_l, batch_b, batch_r, batch_y = batch_loader
        batch_l, batch_b, batch_r, batch_y = batch_l.to(device), batch_b.to(device), batch_r.to(device), batch_y.to(device)

        model.eval()
        with torch.no_grad():
            pred = model(batch_l.float(), batch_b.float(), batch_r.float())
            loss = loss_fn(pred, batch_y)
            loss = loss.to('cpu')
            avg_loss += loss.item()
            
        pred_y = torch.max(pred, 1).indices
        total_pred.append(pred_y.cpu())
        total_y.append(batch_y.cpu())
    
    avg_loss = avg_loss / (batch_i+1)
    
    total_y = torch.cat(total_y)
    total_pred = torch.cat(total_pred)
    acc = accuracy_score(total_y, total_pred)
    micro_f1 = f1_score(total_y.cpu(), total_pred.cpu(), average='micro')
    macro_f1 = f1_score(total_y.cpu(), total_pred.cpu(), average='macro')

    print(f'avg loss: {avg_loss:.4f}')
    print(f'acc: {acc:.4f}')
    print(f'micro_f1: {micro_f1:.4f}')
    print(f'macro_f1: {macro_f1:.4f}')