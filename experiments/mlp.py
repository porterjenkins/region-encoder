import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


class MLP(nn.Module):
    def __init__(self, n_features, h_dim):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(n_features, h_dim)
        self.hidden_layer = nn.Linear(h_dim, 1)
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, X):
        H = self.input_layer(X)
        y_hat = self.hidden_layer(H)

        return y_hat

    def get_optimizer(self, lr):
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        return optimizer

    def loss_l2_penalty(self, y_hat, y_true, lmbda):
        # MSE with l2 norm penalty on weights (Lasso)
        err = y_hat - y_true
        mse = torch.mean(torch.pow(err, 2))

        reg = 0

        for param in self.parameters():

            reg += lmbda*torch.norm(param)

        loss = mse + reg

        return loss



    def fit(self, X, y, n_epochs, eta, batch_size, lmbda, verbose=True,  X_val=None, y_val=None):
        optimizer = self.get_optimizer(eta)
        n_samples = X.shape[0]
        self.loss_seq = list()
        self.loss_seq_val = list()



        for epoch_cntr in range(n_epochs):
            data_idx = np.random.permutation(np.arange(n_samples))

            for step in range(n_samples // batch_size):
                start_idx = step*batch_size
                end_idx = step*batch_size + batch_size
                batch_idx = data_idx[start_idx:end_idx]
                optimizer.zero_grad()

                y_hat = self.forward(X[batch_idx, :])

                loss = self.loss_l2_penalty(y_hat, y[batch_idx], lmbda)
                loss.backward()
                optimizer.step()

            if X_val is not None and y_val is not None:

                y_hat_val = self.forward(X_val)
                loss_val = self.loss_l2_penalty(y_hat_val, y_val, lmbda)

                if verbose:
                    print("Epoch: {}, Train Loss: {:.4f}, Val. loss: {:.4f}".format(epoch_cntr, loss.item(), loss_val.item()))
            else:
                print("Epoch: {}, Train Loss: {:.4f}".format(epoch_cntr, loss.item()))

            self.loss_seq.append(loss.item())
            self.loss_seq_val.append(loss_val.item())

    def predict(self, X):
        return self.forward(X)




    def plot_learning_curve(self, fname):
        x = np.arange(len(self.loss_seq))

        plt.plot(x, self.loss_seq, label='training loss')
        plt.plot(x, self.loss_seq_val, label='validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc='best')
        plt.savefig(fname)
        plt.clf()
        plt.close()



if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data('data')

    lambda_vals = [0, .01, .1, 1, 4, 8, 16, 32]
    errs = list()

    for l in lambda_vals:
        mod = MLP(n_features=X_train.shape[1], h_dim=64)
        mod.run_train_job(X_train, y_train, X_val, y_val, 100, 1e-4, 100, lmbda=l, verbose=False)
        y_hat_val = mod.forward(X_val)
        val_mse = mean_squared_error(y_val.detach().numpy(), y_hat_val.detach().numpy())
        print("Lambda: {}, Validation mse: {:.4f}".format(l, val_mse))
        errs.append(val_mse)

    keep_lambda = lambda_vals[np.argmin(errs)]
    print("Best Lambda: {}".format(keep_lambda))
