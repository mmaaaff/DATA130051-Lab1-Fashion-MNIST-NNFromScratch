import os
import MyDL
from MyDL.data import *

def train(model, criterion, optimizer, train_data, val_data, num_epochs=10,
           batch_size=256, lambda_L2=0.001, path='model_params', 
           continue_if_exists=False):
    model_name = 'MLP3_({},{})_{}_L2-{}_lr-{}'.format(model.hidden_size1, model.hidden_size2, model.activ_func, lambda_L2, optimizer.lr)
    continued_train = False
    if os.path.exists(f'{path}/{model_name}.npz'):
        print(f"Model already exists. Loading model...")
        model.load(f'{path}/{model_name}.npz')
        if not continue_if_exists:
            print(f"Model loaded successfully.")
        else:
            print(f"Model loaded successfully. Training will be continued.")
        continued_train = True
    if continued_train and not continue_if_exists:
        print('Model is not going to be trained further as continue_if_exists is set to False.\n')
        with np.load(os.path.join('results', f'{model_name}.npz')) as results:
            train_loss = results['train_loss'].tolist()
            val_loss = results['val_loss'].tolist()
            train_acc = results['train_acc'].tolist()
            val_acc = results['val_acc'].tolist()
        return train_loss, val_loss, train_acc, val_acc, continued_train
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(num_epochs):
        model.train()
        epoch_training_loss = 0.0
        correct = 0
        train_loader = Dataloader(train_data, batch_size, shuffle=True)
        for i, (X_batch, y_batch) in enumerate(train_loader):
            output = model(X_batch)
            loss = criterion(output, y_batch)
            L2 = MyDL.MyTensor(0.)
            for param in model.params:
                L2 = L2 + param.square().sum().item()
            loss_with_L2 = loss + lambda_L2 * L2
            epoch_training_loss += loss.data * len(X_batch)
            y_pred = output.data.argmax(axis=1)
            correct += (y_pred == y_batch.data).sum()
            optimizer.zero_grad()
            loss_with_L2.backward()
            optimizer.step()
        epoch_training_loss /= len(train_data)
        acc = correct / len(train_data)
        train_loss.append(epoch_training_loss)
        train_acc.append(acc)
        print(f"Epoch {epoch + 1}/{num_epochs}. Training Loss:   {epoch_training_loss:.3f} \t Accuracy: {acc:.3f}")
        epoch_val_loss = 0.0
        correct = 0.0
        val_loader = Dataloader(val_data, batch_size, shuffle=False)
        if val_data is not None:
            model.eval()
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                loss = criterion(output, y_batch)
                y_pred = output.data.argmax(axis=1)
                correct += (y_pred == y_batch.data).sum()
                epoch_val_loss += loss.data * len(X_batch)
            epoch_val_loss /= len(val_data)
            acc = correct / len(val_data)
            val_loss.append(epoch_val_loss)
            val_acc.append(acc)
            spaces = len(f'Epoch {epoch + 1}/{num_epochs}.') * ' '
            print(f"{spaces} Validation Loss: {epoch_val_loss:.3f} \t Accuracy: {acc:.3f}")
    model.save(filename=f'{model_name}.npz', path=path)
    print('\n')
    return train_loss, val_loss, train_acc, val_acc, continued_train


def test(model, test_data, batch_size=256):
    model.eval()
    correct = 0
    test_loader = Dataloader(test_data, batch_size, shuffle=False)
    for X_batch, y_batch in test_loader:
        output = model(X_batch)
        y_pred = output.data.argmax(axis=1)
        correct += (y_pred == y_batch.data).sum()
    acc = correct / len(test_data)
    return acc


def save_result(train_loss, val_loss, train_acc, val_acc, model_name, continued_train='false', path='results'):
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f'{model_name}.npz'
    path = os.path.join(path, filename)
    train_loss_arr = np.array(train_loss)
    val_loss_arr = np.array(val_loss)
    train_acc_arr = np.array(train_acc)
    val_acc_arr = np.array(val_acc)
    if continued_train:
        prev_results = np.load(path)
        train_loss_arr = np.concatenate((prev_results['train_loss'], train_loss_arr))
        val_loss_arr = np.concatenate((prev_results['val_loss'], val_loss_arr))
        train_acc_arr = np.concatenate((prev_results['train_acc'], train_acc_arr))
        val_acc_arr = np.concatenate((prev_results['val_acc'], val_acc_arr))
    np.savez(path, train_loss=train_loss_arr, val_loss=val_loss_arr, train_acc=train_acc_arr, val_acc=val_acc_arr)


def load_result(model_name, path='results'):
    filename = f'{model_name}.npz'
    path = os.path.join(path, filename)
    with np.load(path) as results:
        train_loss = results['train_loss']
        val_loss = results['val_loss']
        train_acc = results['train_acc']
        val_acc = results['val_acc']
    return train_loss, val_loss, train_acc, val_acc