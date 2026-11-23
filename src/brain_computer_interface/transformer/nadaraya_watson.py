import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import torch
    from torch import nn
    import matplotlib.pyplot as plt

    return nn, plt, torch


@app.cell
def _(torch):
    n_train = 50  # 训练样本数
    x_train, _ = torch.sort(torch.rand(n_train) * 5)  # 排序后的训练样本


    def f(x):
        return 2 * torch.sin(x) + x**0.8


    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
    x_test = torch.arange(0, 5, 0.1)  # 测试样本
    y_truth = f(x_test)  # 测试样本的真实输出
    n_test = len(x_test)  # 测试样本数
    n_test
    return n_test, n_train, x_test, x_train, y_train, y_truth


@app.function
def plot_kernel_reg(x_test, x_train, y_train, y_truth, y_hat, plt):
    plt.figure(figsize=(5, 2.7), layout="constrained")
    plt.plot(
        x_test,
        y_truth,
    )
    plt.plot(x_train, y_train, "o", alpha=0.5)
    plt.plot(
        x_test,
        y_hat,
    )
    plt.show()


@app.cell
def _(n_test, plt, torch, x_test, x_train, y_train, y_truth):
    y_hat = torch.repeat_interleave(y_train.mean(), n_test)
    plot_kernel_reg(x_test, x_train, y_train, y_truth, y_hat, plt)
    return


@app.cell
def _(n_train, nn, plt, torch, x_test, x_train, y_train, y_truth):
    # X_repeat的形状:(n_test,n_train),
    # 每一行都包含着相同的测试输入（例如：同样的查询）
    X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    # x_train包含着键。attention_weights的形状：(n_test,n_train),
    # 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
    attention_weights = nn.functional.softmax(-((X_repeat - x_train) ** 2) / 2, dim=1)
    # y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
    y_hat1 = torch.matmul(attention_weights, y_train)
    plot_kernel_reg(x_test, x_train, y_train, y_truth, y_hat1, plt)
    return (attention_weights,)


@app.cell
def _(attention_weights):
    from brain_computer_interface.transformer.utils import show_heatmaps

    show_heatmaps(
        attention_weights.unsqueeze(0).unsqueeze(0),
        xlabel="Sorted training inputs",
        ylabel="Sorted testing inputs",
    )
    return


@app.cell
def _(torch):
    weights = torch.ones((2, 10)) * 0.1
    values = torch.arange(20.0).reshape((2, 10))
    torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))

    (
        weights.unsqueeze(1),
        values.unsqueeze(-1),
        torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1)),
    )
    return


@app.cell
def _(nn, torch):
    class NWKernelRegression(nn.Module):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

        def forward(self, queries, keys, values):
            # queries和attention_weights的形状为(查询个数，“键－值”对个数)
            queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
            self.attention_weights = nn.functional.softmax(
                -(((queries - keys) * self.w) ** 2) / 2, dim=1
            )
            # values的形状为(查询个数，“键－值”对个数)
            return torch.bmm(
                self.attention_weights.unsqueeze(1), values.unsqueeze(-1)
            ).reshape(-1)

    return (NWKernelRegression,)


@app.cell
def _(n_train, torch, x_train, y_train):
    # X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
    X_tile = x_train.repeat((n_train, 1))
    # Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
    Y_tile = y_train.repeat((n_train, 1))
    # keys的形状:('n_train'，'n_train'-1)
    keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    # values的形状:('n_train'，'n_train'-1)
    values1 = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    return keys, values1


@app.cell
def _(NWKernelRegression, keys, nn, torch, values1, x_train, y_train):
    net = NWKernelRegression()
    loss = nn.MSELoss(reduction="none")
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    # animator = d2l.Animator(xlabel="epoch", ylabel="loss", xlim=[1, 5])

    for epoch in range(5):
        trainer.zero_grad()
        l = loss(net(x_train, keys, values1), y_train)
        l.sum().backward()
        trainer.step()
        print(f"epoch {epoch + 1}, loss {float(l.sum()):.6f}")
        # animator.add(epoch + 1, float(l.sum()))
    return (net,)


@app.cell
def _(keys, n_test, net, plt, values1, x_test, x_train, y_train, y_truth):
    # keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
    keys2 = x_train.repeat((n_test, 1))
    # value的形状:(n_test，n_train)
    values2 = y_train.repeat((n_test, 1))
    y_hat2 = net(x_test, keys, values1).unsqueeze(1).detach()
    plot_kernel_reg(x_test, x_train, y_train, y_truth, y_hat2, plt)
    return


if __name__ == "__main__":
    app.run()
