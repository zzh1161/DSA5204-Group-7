# Code for DSA5204 Project -- Group 7

## Environment
+ Python 3.10
+ Pytorch 2.1.0
+ Cuda 11.8

## Running configuration
+ For each sample, run `generate_data.py` to generate data. Run `main.py` to train the model and do the test.
    ```python
    if __name__ == '__main__':
        # train()
        test()
    ```
+ For `Burger_equation`, change `generate_data.py` for different initial conditions.
    ```python
    def any_solution(x0f,t):
        # u = 0.5 + torch.sin(x0f*2.*math.pi)
        # u = 0.5 + torch.cos(x0f*2.*math.pi)
        # u = 0.5 - torch.sin(x0f*2.*math.pi)
        u = 2.*torch.exp(-10.*x0f**2) - 0.5
        dx = x0f[0,0,1]-x0f[0,0,0]
        ff = CentDif(dx)
        ff.eval()
        u = runge_kutta(u, t, ff, 0.001)
        return u
    ```
+ For `Schrodinger_2D`, we only save the generated gif in the folder. Running the code generates every frame of the simulation.