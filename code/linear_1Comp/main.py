import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import h5py
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))

def to_numpy(x):
    return x.detach().cpu().numpy()

def boundary_condition(u, igst):
    u = torch.cat((u[:, :, -2*igst:-igst], u[:, :, igst: -igst], u[:, :, igst:2*igst]), dim=2)
    return u

def plot_u(idx, x0, ur, un, ut):
    plt.clf()
    x = to_numpy(torch.squeeze(x0))
    un = to_numpy(un)
    ut = to_numpy(ut)
    ur = to_numpy(ur)
    plt.clf()
    plt.scatter(x, ur, c='g', marker='o', label='Roe solver', alpha=0.3)
    plt.scatter(x, un, c='r', marker='s', label='RoeNet', alpha=0.3)
    plt.plot(x, ut, label='ground truth')
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.draw()
    plt.savefig('figs/linear1c_{}.png'.format(idx))
    plt.pause(0.01)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_type):
        f = h5py.File('data.h5')
        self.uC0 = f['uC0'][:]
        self.uC1 = f['uC1'][:]
        self.DT = f['DT'][:]
        split = int(self.uC0.shape[0] * 0.9)
        if data_type == 'train':
            self.uC0, self.uC1, self.DT = self.uC0[:split], self.uC1[:split], self.DT[:split]
        else:
            self.uC0, self.uC1, self.DT = self.uC0[split:], self.uC1[split:], self.DT[split:]
        f.close()

    def __getitem__(self, index):
        return self.uC0[index], self.uC1[index], self.DT[index]

    def __len__(self):
        return self.uC0.shape[0]

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=3):
        super(ResBlock, self).__init__()
        self.kernel_size = kernel_size
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=1, padding=0),
        )
        self.shortcut = nn.Sequential()
        if inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, padding=0),
            )

    def forward(self, x):
        if self.kernel_size == 3:
            out = self.left(F.pad(x, pad=[1, 1], mode='replicate'))
        else:
            out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Roenet(nn.Module):
    def __init__(self, dx, igst, eps):
        super(Roenet, self).__init__()
        self.component = 1
        self.hide = 1
        self.igst = igst
        self.dx = dx
        self.eps = eps

        self.cal_lamda = nn.Sequential(ResBlock(self.component*2, 16),
                                      ResBlock(16, 16),
                                      ResBlock(16, 32),
                                      ResBlock(32, 64),
                                      ResBlock(64, 64),
                                      ResBlock(64, 64, kernel_size=1),
                                      nn.Conv1d(64, self.hide, kernel_size=1, padding=0),
                                      )
        self.cal_L = nn.Sequential(ResBlock(self.component*2, 16),
                                   ResBlock(16, 16),
                                   ResBlock(16, 32),
                                   ResBlock(32, 64),
                                   ResBlock(64, 64),
                                   ResBlock(64, 64, kernel_size=1),
                                   nn.Conv1d(64, self.component*self.hide, kernel_size=1, padding=0),
                                   )

    def get_du(self, um):
        ul = torch.cat((um[:, :, :1], um[:, :, :-1]), 2)
        ur = torch.cat((um[:, :, 1:], um[:, :, -1:]), 2)
        data_left = torch.cat((ul, um), 1)
        data_right = torch.cat((um, ur), 1)

        lamda_l = self.cal_lamda(data_left).transpose(1, 2) / 10
        lamda_l = torch.diag_embed(lamda_l)
        lamda_r = self.cal_lamda(data_right).transpose(1, 2) / 10
        lamda_r = torch.diag_embed(lamda_r)
        L_l = self.cal_L(data_left).transpose(1, 2)
        L_l = L_l.reshape(L_l.shape[0], L_l.shape[1], self.hide, self.component)
        L_r = self.cal_L(data_right).transpose(1, 2)
        L_r = L_r.reshape(L_r.shape[0], L_r.shape[1], self.hide, self.component)
        R_l = torch.inverse((L_l.transpose(2,3))@L_l)@(L_l.transpose(2,3))
        R_r = torch.inverse((L_r.transpose(2,3))@L_r)@(L_r.transpose(2,3))

        um = um.transpose(1, 2).unsqueeze(-1)
        ul = ul.transpose(1, 2).unsqueeze(-1)
        ur = ur.transpose(1, 2).unsqueeze(-1)

        out = R_r @ (lamda_r - lamda_r.abs()) @ L_r @ (ur - um) + \
              R_l @ (lamda_l + lamda_l.abs()) @ L_l @ (um - ul)
        return -out.squeeze(-1).transpose(1, 2) / (2 * self.dx)

    def forward(self, z0, Delta_t):
        steps = round(Delta_t / self.eps)
        h = Delta_t / steps
        z = z0
        for _ in range(int(steps)):
            z = boundary_condition(z + h * self.get_du(z), self.igst)
        return z

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
igst = 10
grid_size = 100
xs = -0.5
xe = 0.5
lx = xe - xs
dx = lx / grid_size
DT = 0.02
model = Roenet(dx, igst, eps=0.01)
model.to(device)
x0 = torch.tensor(range(grid_size), dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0) * lx / grid_size + xs
m, n, l = x0.shape
x0f = torch.zeros(m, n, l + igst * 2)
x0f[:, :, 0:igst] = x0[:, :, l - igst:l] - lx
x0f[:, :, l + igst:l + igst * 2] = x0[:, :, 0:igst] + lx
x0f[:, :, igst:l + igst] = x0[:, :, 0:l]
time_size = 101

def train():
    epochs = 500
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    train_data_loader = torch.utils.data.DataLoader(Dataset('train'), batch_size=32, shuffle=True)
    for i in range(epochs):
        train_loss_mse = 0
        train_sample = 0
        model.train()
        for _, data_batch in enumerate(train_data_loader):
            uC0, uC1, DT = data_batch
            uC0, uC1 = uC0.to(device), uC1.to(device)
            uN1 = model(uC0, DT[0].cpu().item())
            lossmse = F.mse_loss(uN1, uC1)
            if(i>0):
                optimizer.zero_grad()
                lossmse.backward()
                optimizer.step()
            train_loss_mse += lossmse.detach().cpu().item()
            train_sample += 1
        scheduler.step()
        print(i, train_loss_mse / train_sample)    
        torch.save(model.state_dict(), "model.pt")

def test():
    model.load_state_dict(torch.load("model.pt"))
    model.to(device)
    model.eval()
    uR = torch.load('uRoe.dat')
    uA = torch.load('uAny.dat')
    uF = uR[0:1, :, :].to(device)
    dim = 0
    with torch.no_grad():
        for i in range(time_size):
            if i == 0 or i == 20 or i == 40 or i == 60:
                plot_u(i, x0f, uR[i, dim, :], uF[:, dim, :], uA[i, dim, :])
            uF = model(uF, DT)

if __name__ == '__main__':
    # train()
    test()