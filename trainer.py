from torch.nn.functional import normalize
from argparser import TRAIN_DEVICE
from tqdm import tqdm, trange
from dataGenerator import *
from model import *
from Data import *


import torch
import torch.nn as nn
import torch.optim as optim

L1Loss = nn.L1Loss()
L2Loss = nn.MSELoss()

class Trainer(object):

    def __init__(self, model, data_generator, args):
        self.model = model.to(TRAIN_DEVICE)
        self.data_generator = data_generator
        self.args = args
        self.loss = nn.MSELoss()
        self.optim = optim.Adam(self.model.parameters(), lr=0.01)

    def load_best_model(self):
        self.model.load_state_dict(torch.load("saved_models/best_model"))        

    def train(self):
        self.model.train()
        torch.autograd.set_detect_anomaly(True)
        (train_R, train_C, train_u_in, train_u_out), train_label = self.data_generator.get_train_features()
        (test_R, test_C, test_u_in, test_u_out), test_label = self.data_generator.get_vali_features()
        best_train_loss = 10000
        best_vali_loss = 10000
        scheduler = optim.lr_scheduler.MultiStepLR(self.optim,milestones=[50,300],gamma = 0.3)
        with trange(self.args.epoch) as progress:
            for ep in progress:
                if ep > 5:
                    self.loss = L1Loss
                else:
                    self.loss = L2Loss
                train_loss = self.train_epoch(train_R, train_C, train_u_in, train_u_out, train_label)
                vali_loss = self.eval_epoch(test_R, test_C, test_u_in, test_u_out, test_label)
                print(f'[epoch {ep}] train loss: {train_loss}, vali loss: {vali_loss}')
                scheduler.step()
                if train_loss < best_train_loss and vali_loss < best_vali_loss:
                    best_train_loss = train_loss
                    best_vali_loss = vali_loss
                    torch.save(self.model.state_dict(), "saved_models/best_model")
                    with open("train.log", 'w') as f:
                        f.write(f"best train loss: {train_loss} \nbest vali loss: {vali_loss} \nepoch: {ep}")

    def eval_on_test(self, feature):
        self.model.load_state_dict(torch.load("saved_models/best_model"))
        R, C, u_in, u_out = self.data_generator.get_eval_features(feature)
        R = R
        C = C
        u_in = u_in
        u_out = u_out
        res = []
        with torch.no_grad():
            batch_size = self.args.batch_size
            batch_number = R.shape[0] // batch_size + 1
            for batch_idx in range(batch_number):
                batch_start = batch_size * batch_idx
                batch_end = min(batch_start + batch_size, R.shape[0])
                batch_R = R[batch_start:batch_end].to(TRAIN_DEVICE)
                batch_C = C[batch_start:batch_end].to(TRAIN_DEVICE)
                batch_u_in = u_in[batch_start:batch_end].to(TRAIN_DEVICE)
                batch_u_out = u_out[batch_start:batch_end].to(TRAIN_DEVICE)
                o = self.model(batch_R, batch_C, batch_u_in, batch_u_out)
                res.append(o.cpu().flatten())
        result = torch.cat(res).flatten()
        # result = self.data_generator.inverse_label(result.reshape(-1, 1))
        submit = pd.DataFrame(columns=['id', 'pressure'])
        submit['id'] = range(1, result.shape[0] + 1)
        submit['pressure'] = result
        submit.to_csv('submit.csv', index=False)
    
    def train_epoch(self, R, C, u_in, u_out, label):
        batch_size = self.args.batch_size
        batch_number = R.shape[0] // batch_size + 1
        all_loss = 0.0
        for batch_idx in range(batch_number):
            batch_start = batch_size * batch_idx
            batch_end = min(batch_start + batch_size, R.shape[0])
            batch_R = R[batch_start:batch_end].to(TRAIN_DEVICE)
            batch_C = C[batch_start:batch_end].to(TRAIN_DEVICE)
            batch_u_in = u_in[batch_start:batch_end].to(TRAIN_DEVICE)
            batch_u_out = u_out[batch_start:batch_end].to(TRAIN_DEVICE)
            batch_label = label[batch_start:batch_end].to(TRAIN_DEVICE)
            out = self.model(batch_R, batch_C, batch_u_in, batch_u_out)
            # print(out.flatten().shape, batch_label.flatten().shape)
            loss = self.loss(out, batch_label)
            all_loss = all_loss + loss
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        all_loss /= batch_number
        return all_loss

    def eval_epoch(self, R, C, u_in, u_out, label):
        with torch.no_grad():
            batch_size = self.args.batch_size
            batch_number = R.shape[0] // batch_size + 1
            all_loss = 0.0
            # l1_loss = 0.0
            for batch_idx in range(batch_number):
                batch_start = batch_size * batch_idx
                batch_end = min(batch_start + batch_size, R.shape[0])
                batch_R = R[batch_start:batch_end].to(TRAIN_DEVICE)
                batch_C = C[batch_start:batch_end].to(TRAIN_DEVICE)
                batch_u_in = u_in[batch_start:batch_end].to(TRAIN_DEVICE)
                batch_u_out = u_out[batch_start:batch_end].to(TRAIN_DEVICE)
                batch_label = label[batch_start:batch_end].to(TRAIN_DEVICE)
                o = self.model(batch_R, batch_C, batch_u_in, batch_u_out)
                loss = self.loss(o, batch_label)
                # pdb.set_trace()
                # l1_loss = l1_loss + L1Loss(self.data_generator.inverse_label(o.clone().detach().cpu().reshape(-1, 1)), self.data_generator.inverse_label(batch_label.clone().detach().cpu().reshape(-1, 1)))
                all_loss = all_loss + loss
            all_loss /= batch_number
            # l1_loss /= batch_number
            return all_loss

if __name__ == "__main__":
    file = 'train.csv'
    vdp = VenDataProcesser()
    vdp.read(file)
    # vdp.summary()
    feature, label = vdp.default_process(split_dataset=True, split_ratio=0.8, vali_set=True, eval=False)

    dg = DataGenerator(feature, label)
    model = Model_V2(128, 1, 128)
    
    trainer = Trainer(model, dg, args)
    trainer.train()

    test_file = 'test.csv'
    vdp.read(test_file)
    feature = vdp.default_process(split_dataset=True, split_ratio=0.8, vali_set=True, eval=True)
    trainer.eval(feature)
