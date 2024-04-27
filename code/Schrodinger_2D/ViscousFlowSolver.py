import torch
import torch.nn as nn
import numpy as np

class viscous_flow_solver_2d(nn.Module):
    def __init__(
            self,
            n_grid = [512,512],
            range_x = [-torch.pi,torch.pi],
            range_y = [-torch.pi,torch.pi],
            nu = 4e-4,
            dt = 0.03
            ):
        super(viscous_flow_solver_2d, self).__init__()
        grid_x = n_grid[0]; grid_y = n_grid[1]
        xs = range_x[0]; xe = range_x[1]
        ys = range_y[0]; ye = range_y[1]
        lx = xe - xs; ly = ye - ys
        dx = lx / grid_x; dy = ly / grid_y

        x = torch.tensor(range(grid_x), dtype=torch.float32)*dx + xs
        y = torch.tensor(range(grid_y), dtype=torch.float32)*dy + ys
        mesh_x, mesh_y = torch.meshgrid(x, y)
        # mesh_x = mesh_x.unsqueeze(0).unsqueeze(0)
        # mesh_y = mesh_y.unsqueeze(0).unsqueeze(0)

        a = torch.tensor(range(grid_x), dtype=torch.float32) + grid_x/2
        b = torch.tensor(range(grid_y), dtype=torch.float32) + grid_y/2
        a = complex(0,1) * (a % grid_x - grid_x/2)
        b = complex(0,1) * (b % grid_y - grid_y/2)
        kx, ky = torch.meshgrid(a, b)
        k2 = kx*kx + ky*ky
        k2[0,0] = -1
        # kx = kx.unsqueeze(0).unsqueeze(0)
        # ky = ky.unsqueeze(0).unsqueeze(0)
        # k2 = k2.unsqueeze(0).unsqueeze(0)

        self.mesh_x = nn.Parameter(mesh_x, requires_grad=False)
        self.mesh_y = nn.Parameter(mesh_y, requires_grad=False)
        self.kx = nn.Parameter(kx, requires_grad=False)
        self.ky = nn.Parameter(ky, requires_grad=False)
        self.k2 = nn.Parameter(k2, requires_grad=False)
        self.dt = dt
        self.nu = nu

    def advection(self, ux, uy):
        wz = torch.fft.ifft2(self.kx*torch.fft.fft2(uy)-self.ky*torch.fft.fft2(ux))
        return uy*wz,-ux*wz
    
    def forward(self, ux, uy, Delta_t):
        sub_steps = round(Delta_t / self.dt)
        h = Delta_t / sub_steps
        for _ in range(int(sub_steps)):
            k1x, k1y = self.advection(ux,uy)
            k2x, k2y = self.advection(ux+0.5*k1x*h,uy+0.5*k1y*h)
            k3x, k3y = self.advection(ux+0.5*k2x*h,uy+0.5*k2y*h)
            k4x, k4y = self.advection(ux+k3x*h,uy+k3y*h)
            ux = ux + ((k1x+2*k2x+2*k3x+k4x)/6 + 0.025*torch.sin(2*self.mesh_x)*torch.cos(2*self.mesh_y))*h
            uy = uy + ((k1y+2*k2y+2*k3y+k4y)/6 - 0.025*torch.cos(2*self.mesh_x)*torch.sin(2*self.mesh_y))*h
            phi = (torch.fft.fft2(ux)*self.kx + torch.fft.fft2(uy)*self.ky) / self.k2
            ux = torch.fft.ifft2(torch.fft.fft2(ux - torch.fft.ifft2(phi*self.kx))*torch.exp(self.nu*self.k2*h))
            uy = torch.fft.ifft2(torch.fft.fft2(uy - torch.fft.ifft2(phi*self.ky))*torch.exp(self.nu*self.k2*h))
        p = torch.fft.ifft2(self.kx*torch.fft.fft2(uy)-self.ky*torch.fft.fft2(ux))
        p = torch.fft.ifft2((self.kx*torch.fft.fft2(uy*p)-self.ky*torch.fft.fft2(ux*p))/self.k2)-0.5*(ux*ux+uy*uy)
        return ux.real, uy.real, p.real
    
    def vel_to_vor(self, ux, uy):
        return torch.fft.ifft2(self.kx*torch.fft.fft2(uy)-self.ky*torch.fft.fft2(ux)).real