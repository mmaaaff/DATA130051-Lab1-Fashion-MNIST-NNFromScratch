import os
import MyDL
from MyDL.data import *
from MyDL import nn


class runner():
    def __init__(self, model:nn.NeuralNetwork, model_name:str, optimizer:nn.Optimizer, criterion, batch_size:int, scheduler:nn.Scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = criterion
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.model_name = model_name

    def train(self, train_data, val_data, num_epochs=10, lambda_L2=0.001, 
            model_path='model_params', result_path='results',
            continue_if_exists=False, val_interval=10, print_interval=50):
        
        continued_train = False

        # Check whether model exsists
        if os.path.exists(f'{model_path}/{self.model_name}.npz'):
            print(f"Model already exists. Loading model from {model_path}/{self.model_name}.npz...")
            self.model.load(f'{model_path}/{self.model_name}.npz')
            result = np.load(f'{result_path}/{self.model_name}.npz', allow_pickle=True) if np.__name__ == 'numpy' else np.load(f'{result_path}/{self.model_name}.npz', allow_pickle=True).npz_file
            result = {key: result[key] for key in result}
            if continue_if_exists:
                continued_train = True
                print(f"Model loaded successfully. Training will be continued.")
                train_loss_iter, val_loss_iter, train_acc_iter, val_acc_iter = result['train_loss_iter'].tolist(), result['val_loss_iter'].tolist(), result['train_acc_iter'].tolist(), result['val_acc_iter'].tolist()
                train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch = result['train_loss_epoch'].tolist(), result['val_loss_epoch'].tolist(), result['train_acc_epoch'].tolist(), result['val_acc_epoch'].tolist()
                batch_size_till_iter = result['batch_size_till_iter'].tolist()
                if val_interval == 0:
                    # Abandon the previous fine grained validation loss and accuracy
                    val_loss_iter = []
                    val_acc_iter = []
            else:
                print(f"Model loaded successfully.")
                print('Model is not going to be trained further as continue_if_exists is set to False.\n')
                return result
        else:
            train_loss_iter, val_loss_iter, train_acc_iter, val_acc_iter = [], [], [], []
            train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch = [], [], [], []
            batch_size_till_iter = []

        if len(val_loss_iter) == 0 and continued_train:
            val_interval = 0
            print('Previous fine grained validation loss and accuracy are not available. val_interval is thus set to 0 to disable fine grained validation loss calculation.')

        self.model.train()
        for epoch in range(num_epochs):
            self.model.train()
            epoch_training_loss = 0.0
            correct = 0
            train_loader = Dataloader(train_data, self.batch_size, shuffle=True)
            for i, (X_batch, y_batch) in enumerate(train_loader):
                output = self.model(X_batch)
                loss = self.loss_fn(output, y_batch)
                train_loss_iter.append(loss.data)
                L2 = MyDL.MyTensor(0.)
                n = 0
                for param in self.model.params:
                    if param.requires_grad:
                        L2 = L2 + param.square().sum().item()
                        n += np.sum(np.ones_like(param.data)).item()
                L2 = L2 * (1 / n)
                loss_with_L2 = loss + lambda_L2 * L2
                epoch_training_loss += loss.data * len(X_batch)
                y_pred = output.data.argmax(axis=1)
                iter_correct = (y_pred == y_batch.data).sum()
                correct += iter_correct
                iter_acc = iter_correct / len(X_batch)
                train_acc_iter.append(iter_acc)
                self.optimizer.zero_grad()
                loss_with_L2.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                if val_interval > 0 and ((i + 1) % val_interval == 0):
                    val_loss, val_acc = self.eval(val_data, 2 * self.batch_size)  # Larger batch size for validation as it doesn't require backpropagation. Boosts speed.
                    val_loss_iter.append(val_loss)
                    val_acc_iter.append(val_acc)
                if (i + 1) % print_interval == 0:
                    print(f"iter {i+1}\t\t loss {loss.data:.3f}")
            epoch_training_loss /= len(train_data)
            epoch_training_acc = correct / len(train_data)
            train_loss_epoch.append(epoch_training_loss)
            train_acc_epoch.append(epoch_training_acc)
            print(f"Epoch {epoch + 1}/{num_epochs}. Training Loss:   {epoch_training_loss:.3f} \t Accuracy: {epoch_training_acc:.3f}")
            val_loss, val_acc = self.eval(val_data, 2 * self.batch_size)
            val_loss_epoch.append(val_loss)
            val_acc_epoch.append(val_acc)
            spaces = len(f'Epoch {epoch + 1}/{num_epochs}.') * ' '
            print(f"{spaces} Validation Loss: {val_loss:.3f} \t Accuracy: {val_acc:.3f}")
        self.model.save(filename=f'{self.model_name}.npz', path=model_path)
        print('\n')

        batch_size_till_iter.append([self.batch_size, len(train_loss_iter)])

        print(len(val_loss_iter))
        result = {'train_loss_iter': train_loss_iter, 'val_loss_iter': val_loss_iter, 'train_acc_iter': train_acc_iter, 'val_acc_iter': val_acc_iter, 'val_interval': val_interval, 'train_loss_epoch': train_loss_epoch, 'val_loss_epoch': val_loss_epoch, 'train_acc_epoch': train_acc_epoch, 'val_acc_epoch': val_acc_epoch, 'continued_train': continued_train, 'batch_size': self.batch_size, 'model_name': self.model_name, 'model_path': model_path, 'batch_size_till_iter': batch_size_till_iter}

        self.save_result(result, result_path)
        return result

    def eval(self, eval_data, batch_size):
        self.model.eval()
        correct = 0
        loss = 0.0
        eval_loader = Dataloader(eval_data, batch_size, shuffle=False)
        for X_batch, y_batch in eval_loader:
            output = self.model(X_batch)
            y_pred = output.data.argmax(axis=1)
            correct += (y_pred == y_batch.data).sum()
            loss += self.loss_fn(output, y_batch).data * len(X_batch)
        acc = correct / len(eval_data)
        loss /= len(eval_data)
        self.model.train()
        return loss, acc

    def save_result(self, result_dict, path):
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f'{self.model_name}.npz'
        path = os.path.join(path, filename)

        result_dict_array = {key: np.array(value) for key, value in result_dict.items() if key not in ['model_name', 'model_path']}

        np.savez(path, **result_dict_array)
        print(f"Results saved to {path}.")
        return

    @staticmethod
    def load_result(model_name, path='results'):
        filename = f'{model_name}.npz'
        path = os.path.join(path, filename)
        with np.load(path, allow_pickle=True) as results:
            return results