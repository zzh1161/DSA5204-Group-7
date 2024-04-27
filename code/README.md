# Code for DSA5204 Project -- Group 7

## Environment
+ Python 3.10
+ Pytorch 2.1.0
+ Cuda 11.8

## Running configuration
+ For each sample, run `generate_data.py` to generate data. Run `main.py` to train the model and do the test. We have trained the model and saved the weights. You don't have to retrain it.
    ```python
    if __name__ == '__main__':
        # train()
        test()
    ```

+ For `Burger_equation`, change `generate_data.py` for different initial conditions. No need to retrain the model after choosing a different initial value.
    ```python
    def any_solution(x0f,t):
        u = 0.5 + torch.sin(x0f*2.*math.pi) # original initial value
        # u = 0.5 + torch.cos(x0f*2.*math.pi) # initial value 1
        # u = 0.5 - torch.sin(x0f*2.*math.pi) # initial value 2
        # u = 2.*torch.exp(-10.*x0f**2) - 0.5 # initial value 3
        dx = x0f[0,0,1]-x0f[0,0,0]
        ff = CentDif(dx)
        ff.eval()
        u = runge_kutta(u, t, ff, 0.001)
        return u
    ```

+ For `Schrodinger_2D`, no need to generate data. Just run `Main.py`, you can get the results. We only save the generated gif in the folder. Running the code generates every frame of the simulation.

+ For `network_test`, three files named with `train` are corresponding to the training codes for the three different networks. Running the three files gives three different models. In the file `error_plot`, we compare the error for the three networks. Running it gives the error comparation plot.

## Acknowledege

Reference was made to the open source code of the original authors of the paper [[repo]](https://github.com/ShiyingXiong/RoeNet), as well as to torchvision’s open source code of resnet [[repo]](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).
