import torch
def tesnordot():
    x = torch.tensor([
        [[1, 2, 1, 1],
         [3, 4, 4, 2]],
        [[2, 2, 1, 1],
         [3, 5, 4, 2]]])
    y = torch.tensor(
        [[0,1],[1,1]])
    print(f"x shape: {x.size()}, y shape: {y.size()}")
    print(x[[[0,1],[1,1]]])
tesnordot()
