import torch

# @torch.jit.script
# def binaryBarrier(x, x_min):
#     if x<= x_min:
#         return torch.tensor([1.],dtype=torch.float)
#     else:
#         return torch.exp(-(x - x_min)*scale) # the higher constant the closer we allow the robots


@torch.jit.script
# def exp(x, scale, x_min):
#     if x<= x_min:
#         return torch.tensor([1.],dtype=torch.float)
#     else:
#         return torch.exp(-(x - x_min)*scale) # the higher constant the closer we allow the robots
@torch.jit.script
def expC(x, scale, x_min):
    # Max capped exponential
    return torch.min(torch.exp(-(x - x_min) * scale), torch.tensor([1.], dtype=torch.float))

# @torch.jit.script
# def expCC(x, scale, x_min, x_max):
# 	# Max and min capped exponential
#     if x<= x_min:
#         return torch.tensor([1.],dtype=torch.float)
#     elif x>x_min and x<=x_max:
#         return torch.exp(-(x - x_min)*scale) # the higher constant the closer we allow the robots
#     else:
#         return torch.tensor([0.],dtype=torch.float)


@torch.jit.script
def expCC(x, scale, x_min, x_max):
    if x <= x_max:
        return torch.min(torch.exp(-(x - x_min) * scale), torch.tensor([1.], dtype=torch.float))
    else:
        return torch.tensor([0.], dtype=torch.float)


@torch.jit.script
def linear(x, x_min, x_max):
    if x <= x_min:
        return torch.tensor([1.], dtype=torch.float)
    elif x > x_min and x <= x_max:
        return -1 / (x_max - x_min) * (x - x_max)
    else:
        return torch.tensor([0.], dtype=torch.float)


@torch.jit.script
def ReLU1F(x, x_min, x_max):
        # Same as linearBarrier but inspired by the ReLU6 function:
        # https://pytorch.org/docs/master/generated/torch.nn.ReLU6.html
        # just flipped around x=(x_max-x_min)/2
    x = -1 / (x_max - x_min) * (x - x_max)
    return torch.min(torch.max(torch.tensor([0]), x), torch.tensor([1]))


@torch.jit.script
def sigmoidS(x, scale, x_min, x_max):
    # shifted sigmoid
    x = x - (x_max - x_min) / 2 - x_min
    return torch.sigmoid(-x * scale)  # the higher constant the closer we allow the robots


@torch.jit.script
def sigmoidSS(x, scale, x_min, x_max):
    # Scaled and shifted sigmoid
    # https://math.stackexchange.com/questions/1214167/how-to-remodel-sigmoid-function-so-as-to-move-stretch-enlarge-it
    # this one is invariant to distance between min and max
    # compare sigmoidS and sigmoidSS with:
    #	x_min = torch.tensor([0.1],dtype=torch.float)
    # 	x_max = torch.tensor([0.5],dtype=torch.float)
    # and e.g.
    #	x_max = torch.tensor([1.5],dtype=torch.float)
    x = x - (x_max - x_min) / 2 - x_min
    A = 1 / (x_max - x_min) / 2
    k = A * torch.log(2 + torch.sqrt(3))
    return torch.sigmoid(-scale * k * x)  # the higher constant the closer we allow the robots


if __name__ == '__main__':
    # only for main
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    scale = torch.tensor([50], dtype=torch.float)
    scale2 = torch.tensor([15], dtype=torch.float)
    x_min = torch.tensor([0.0], dtype=torch.float)
    x_max = torch.tensor([1.0], dtype=torch.float)

    x = torch.arange(x_min.item() - 0.02, x_max.item() + 0.02, 0.0001)
    y_expC = torch.zeros_like(x)
    y_expCC = torch.zeros_like(x)
    y_ReLU1F = torch.zeros_like(x)
    y_sigmoidS = torch.zeros_like(x)
    y_sigmoidSS = torch.zeros_like(x)

    for i in range(len(x)):
        y_expC[i] = expC(x[i], scale, x_min)
        y_expCC[i] = expCC(x[i], scale, x_min, x_max)
        y_ReLU1F[i] = ReLU1F(x[i], x_min, x_max)
        y_sigmoidS[i] = sigmoidS(x[i], 1.75 * scale2, x_min, x_max)
        y_sigmoidSS[i] = sigmoidSS(x[i], scale2, x_min, x_max)

    fig, axs = plt.subplots(1, 1)
    plt.plot([0, 0], [-0.1, 1.1], color='black')
    plt.plot([x_min.item(), x_min.item()], [-0.1, 1.1], '--', color='black')
    plt.plot([x_max.item(), x_max.item()], [-0.1, 1.1], '--', color='black')
    plt.plot(x, y_expC, label="expC")
    plt.plot(x, y_expCC, label="expCC")
    plt.plot(x, y_ReLU1F, label="ReLU1F")
    plt.plot(x, y_sigmoidS, label="SigmoidS")
    plt.plot(x, y_sigmoidSS, label="SigmoidSS")

    axs.legend()
    plt.show(block=True)
