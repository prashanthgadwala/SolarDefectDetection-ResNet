import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import os


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch, is_best=False):
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch)
        t.save({'state_dict': self._model.state_dict()}, checkpoint_path)
        if is_best:
            best_checkpoint_path = 'checkpoints/checkpoint_best.ckp'
            t.save({'state_dict': self._model.state_dict()}, best_checkpoint_path)
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        self._optim.zero_grad()
        y_pred = self._model(x)
        loss = self._crit(y_pred, y)
        loss.backward()
        self._optim.step()
        return loss.item()
        
        
    
    def val_test_step(self, x, y):
        y_pred = self._model(x)
        loss = self._crit(y_pred, y)
        return loss.item(), y_pred
        
    def train_epoch(self):
        self._model.train()
        total_loss = 0
        for x, y in tqdm(self._train_dl, desc="Training"):
            if self._cuda:
                x, y = x.cuda(), y.cuda()
            loss = self.train_step(x, y)
            total_loss += loss
        return total_loss / len(self._train_dl)
    
    def val_test(self):
        self._model.eval()
        with t.no_grad():
            total_loss = 0
            all_preds = []
            all_labels = []
            for x, y in tqdm(self._val_test_dl, desc="Validation/Test"):
                if self._cuda:
                    x, y = x.cuda(), y.cuda()
                loss, preds = self.val_test_step(x, y)
                total_loss += loss
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())
            avg_loss = total_loss / len(self._val_test_dl)
            all_preds = t.cat(all_preds)
            all_labels = t.cat(all_labels)
            f1 = f1_score(all_labels.numpy(), (all_preds.numpy() > 0.5).astype(int), average='macro')
            print(f"Validation Loss: {avg_loss:.4f}, F1 Score: {f1:.4f}")
            return avg_loss, f1
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        train_losses = []
        val_losses = []
        val_f1s = []
        best_val_loss = float('inf')
        patience_counter = 0
        epoch_counter = 0
        
        while True:
            if epochs > 0 and epoch_counter >= epochs:
                break
            train_loss = self.train_epoch()
            val_loss, val_f1 = self.val_test()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1s.append(val_f1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch_counter)
                patience_counter = 0
            else:
                patience_counter += 1
            if self._early_stopping_patience > 0 and patience_counter >= self._early_stopping_patience:
                print("Early stopping triggered.")
                break
            epoch_counter += 1
        return train_losses, val_losses, val_f1s
