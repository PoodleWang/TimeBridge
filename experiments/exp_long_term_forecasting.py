from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        if self.args.data == 'PEMS':
            return nn.L1Loss()
        return nn.MSELoss()

    def time_freq_mae(self, batch_y, outputs):
        # time MAE
        t_loss = (outputs - batch_y).abs().mean()
        # freq MAE
        f_loss = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
        f_loss = f_loss.abs().mean()
        return (1 - self.args.alpha) * t_loss + self.args.alpha * f_loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if self.args.data == 'PEMS':
                    B, T, C = pred.shape
                    pred_np = pred.numpy().reshape(-1, C)
                    true_np = true.numpy().reshape(-1, C)
                    pred_np = vali_data.inverse_transform(pred_np).reshape(B, T, C)
                    true_np = vali_data.inverse_transform(true_np).reshape(B, T, C)
                    mae, mse, rmse, mape, mspe = metric(pred_np, true_np)
                    loss = mae
                else:
                    loss = criterion(pred, true)

                total_loss.append(loss.item() if hasattr(loss, "item") else float(loss))

        self.model.train()
        return float(np.mean(total_loss)) if len(total_loss) else 0.0

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=train_steps,
                pct_start=self.args.pct_start,
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate
            )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = self.time_freq_mae(batch_y, outputs)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / max(1, iter_count)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = float(np.mean(train_loss)) if len(train_loss) else 0.0
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, None, epoch + 1, self.args)
            else:
                try:
                    print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
                except Exception:
                    pass

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        """
        Modified to ALSO export embeddings if model supports return_hidden=True.
        Saves:
          results/<setting>/pred.npy
          results/<setting>/true.npy
          results/<setting>/h_short.npy
          results/<setting>/h_long.npy
        """
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            ckpt = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))

        self.model.eval()
        preds_list = []
        trues_list = []

        # embedding buffers
        h_short_list = []
        h_long_list = []
        embedding_enabled = True

        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # ---- forward (try return_hidden=True) ----
                if self.args.output_attention:
                    # If output_attention, original returns (out, attn). We keep old behavior for preds.
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    hidden = None
                    embedding_enabled = False
                else:
                    if embedding_enabled:
                        try:
                            outputs, hidden = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, return_hidden=True
                            )
                        except TypeError:
                            # Model doesn't accept return_hidden (fallback)
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            hidden = None
                            embedding_enabled = False
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        hidden = None

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                true = batch_y[:, -self.args.pred_len:, f_dim:]

                # ---- collect embeddings ----
                if embedding_enabled and (hidden is not None):
                    # hidden["h_short"] / hidden["h_long"] are [B, C, D]
                    hs = hidden["h_short"].detach().cpu().numpy()
                    hl = hidden["h_long"].detach().cpu().numpy()
                    # Basic numeric guard (avoid breaking save on rare inf/nan)
                    hs = np.nan_to_num(hs, nan=0.0, posinf=0.0, neginf=0.0)
                    hl = np.nan_to_num(hl, nan=0.0, posinf=0.0, neginf=0.0)
                    h_short_list.append(hs)
                    h_long_list.append(hl)

                # ---- to numpy, keep shape (B, pred_len, C) ----
                pred_np = outputs.detach().cpu().numpy()
                true_np = true.detach().cpu().numpy()

                # inverse transform if requested (expects 2D)
                if test_data.scale and self.args.inverse:
                    B, T, C = pred_np.shape
                    pred_np = test_data.inverse_transform(pred_np.reshape(-1, C)).reshape(B, T, C)
                    true_np = test_data.inverse_transform(true_np.reshape(-1, C)).reshape(B, T, C)

                # ensure 3D
                if pred_np.ndim == 2:
                    pred_np = pred_np[:, None, :]
                if true_np.ndim == 2:
                    true_np = true_np[:, None, :]

                preds_list.append(pred_np)
                trues_list.append(true_np)

        if len(preds_list) == 0:
            raise RuntimeError("No predictions collected in test(). Check test_loader.")

        preds = np.concatenate(preds_list, axis=0)   # (T, pred_len, C)
        trues = np.concatenate(trues_list, axis=0)   # (T, pred_len, C)
        print('test shape:', preds.shape, trues.shape)

        # save pred/true
        folder_path = os.path.join("results", setting)
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, "pred.npy"), preds)
        np.save(os.path.join(folder_path, "true.npy"), trues)
        print("Saved preds/trues to:", folder_path)

        # ---- save embeddings (if enabled) ----
        if embedding_enabled and len(h_short_list) > 0:
            h_short = np.concatenate(h_short_list, axis=0)  # (T, C, D)
            h_long = np.concatenate(h_long_list, axis=0)    # (T, C, D)
            np.save(os.path.join(folder_path, "h_short.npy"), h_short)
            np.save(os.path.join(folder_path, "h_long.npy"), h_long)
            print("Saved embeddings to:", folder_path)
            print("h_short shape:", h_short.shape, "h_long shape:", h_long.shape)
        else:
            print("Embedding export skipped (model does not support return_hidden=True or output_attention=True).")

        # metrics
        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds_ = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues_ = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)
        else:
            preds_, trues_ = preds, trues

        mae, mse, rmse, mape, mspe = metric(preds_, trues_)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('rmse:{}, mape:{}, mspe:{}'.format(rmse, mape, mspe))

        # write summary
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
            f.write('\n\n')

        return