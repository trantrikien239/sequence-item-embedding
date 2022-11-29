import torch
from copy import deepcopy

def train_rnn(train_dl, valid_dl, model, model_name, cls_weight_1, cls_weight_2,
    opt, save_path=".", epochs=1, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    loss_func1 = torch.nn.CrossEntropyLoss(weight=torch.tensor(cls_weight_1).to(device))
    loss_func2 = torch.nn.CrossEntropyLoss(weight=torch.tensor(cls_weight_2).to(device))
    model.to(device)
    best_eval_loss = float('inf')
    eval_loss_list = []
    for epoch in range(epochs):
        model.train()
        for (x1b, x2b, sb), yb in train_dl:
            x1b = x1b.to(device)
            x2b = x2b.to(device)
            sb = sb.to(device)
            yb = yb.to(device)
            # print("yb", yb.dtype)
            yb_pred_1, yb_pred_2 = model([x1b, x2b, sb])
            # print("yb_pred_1", yb_pred_1.dtype)

            yb1 = yb[:,0]
            yb2 = yb[:,1]
            loss = loss_func1(yb_pred_1.double(), yb1) + loss_func2(yb_pred_2.double(), yb2)
            loss.backward()
            opt.step()
            opt.zero_grad()
        model.eval()
        with torch.no_grad():
            # total_train_loss = 0
            # for (x1b, x2b, sb), yb in train_dl:
            #     x1b = x1b.to(device)
            #     x2b = x2b.to(device)
            #     sb = sb.to(device)
            #     yb = yb.to(device)
            #     yb_pred_1, yb_pred_2 = model([x1b, x2b, sb])
            #     yb1 = yb[:,0]
            #     yb2 = yb[:,1]
            #     train_loss = loss_func1(yb_pred_1.double(), yb1) + loss_func2(yb_pred_2.double(), yb2)
            #     total_train_loss += train_loss.item()
            total_valid_loss = 0  
            for (x1b, x2b, sb), yb in valid_dl:
                x1b = x1b.to(device)
                x2b = x2b.to(device)
                sb = sb.to(device)
                yb = yb.to(device)
                yb_pred_1, yb_pred_2 = model([x1b, x2b, sb])
                yb1 = yb[:,0]
                yb2 = yb[:,1]
                valid_loss = loss_func1(yb_pred_1.double(), yb1) + loss_func2(yb_pred_2.double(), yb2)
                total_valid_loss += valid_loss.item()
                
        # tl = (total_train_loss / len(train_dl)).cpu().numpy()
        vl = total_valid_loss / len(valid_dl)
        if vl < best_eval_loss:
            best_eval_loss = vl
            best_model = deepcopy(model)
            torch.save(model.state_dict(), f'{save_path}/best_{model_name}.pt')
        eval_loss_list.append(vl)
        print(f"Epoch {epoch:04}, valid loss: {vl:.6f}, best valid loss: {best_eval_loss:.6f}")
        if epoch > 5 and min(eval_loss_list[-10:]) > best_eval_loss:
            print(f"Early stopping, best valid loss: {best_eval_loss:.6f}")
            break
    # best_model = torch.load(f'best_model_{model_name}.pt')
    best_model.eval()
    return best_model