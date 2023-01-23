import torch



# Helper function to train a model
def train_model(model, criterion, optimizer, train_loader, test_loader, epochs, device='cpu'):
    loss_stats = {
        'train': [],
        "val": []
    }

    for e in range(1, epochs + 1):
        # TRAINING
        train_epoch_loss = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()

        # VALIDATION
        with torch.no_grad():
            val_epoch_loss = 0
            model.eval()
            for X_val_batch, y_val_batch in test_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch)
                val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))
                val_epoch_loss += val_loss.item()
        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(test_loader))

        print(
            f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Test Loss: {val_epoch_loss / len(test_loader):.5f}')

    return loss_stats

