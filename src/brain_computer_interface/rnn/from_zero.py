import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import math
    import torch
    from torch import nn
    from torch.nn import functional as F
    from brain_computer_interface.rnn.utils import load_data_time_machine

    return F, load_data_time_machine, nn, torch


@app.cell
def _(load_data_time_machine):
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)   
    return (vocab,)


@app.cell
def _(torch):
    def get_params(vocab_size, num_hiddens, device):
        num_inputs = num_outputs = vocab_size

        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.01

        # 隐藏层参数
        W_xh = normal((num_inputs, num_hiddens))
        W_hh = normal((num_hiddens, num_hiddens))
        b_h = torch.zeros(num_hiddens, device=device)
        # 输出层参数
        W_hq = normal((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)
        # 附加梯度
        params = [W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    return (get_params,)


@app.cell
def _(torch):
    def init_rnn_state(batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device), )

    return (init_rnn_state,)


@app.cell
def _(torch):
    def rnn(inputs, state, params):
        # inputs的形状：(时间步数量，批量大小，词表大小)
        W_xh, W_hh, b_h, W_hq, b_q = params
        H, = state
        outputs = []
        # X的形状：(批量大小，词表大小)
        for X in inputs:
            H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
            Y = torch.mm(H, W_hq) + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H,)

    return (rnn,)


@app.cell
def _(F, torch):
    class RNNModelScratch:  # @save
        """从零开始实现的循环神经网络模型"""

        def __init__(
            self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn
        ):
            self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
            self.params = get_params(vocab_size, num_hiddens, device)
            self.init_state, self.forward_fn = init_state, forward_fn

        def __call__(self, X, state):
            X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
            return self.forward_fn(X, state, self.params)

        def begin_state(self, batch_size, device):
            return self.init_state(batch_size, self.num_hiddens, device)

    return (RNNModelScratch,)


@app.cell
def _(F, RNNModelScratch, get_params, init_rnn_state, rnn, torch, vocab):
    X = torch.arange(10).reshape((2, 5))
    F.one_hot(X.T, 28).shape

    num_hiddens = 512
    net = RNNModelScratch(len(vocab), num_hiddens, "cpu", get_params, init_rnn_state, rnn)
    state = net.begin_state(X.shape[0], "cpu")
    Y, new_state = net(X.to("cpu"), state)
    Y.shape, len(new_state), new_state[0].shape
    return (net,)


@app.cell
def _(torch):
    def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
        """在prefix后面生成新字符"""
        state = net.begin_state(batch_size=1, device=device)
        outputs = [vocab[prefix[0]]]
        get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
        for y in prefix[1:]:  # 预热期
            _, state = net(get_input(), state)
            outputs.append(vocab[y])
        for _ in range(num_preds):  # 预测num_preds步
            y, state = net(get_input(), state)
            outputs.append(int(y.argmax(dim=1).reshape(1)))
        return ''.join([vocab.idx_to_token[i] for i in outputs])

    return (predict_ch8,)


@app.cell
def _(net, predict_ch8, vocab):
    predict_ch8("time traveller ", 10, net, vocab, "cpu")
    return


@app.cell
def _(nn, torch):
    def grad_clipping(net, theta):  # @save
        """裁剪梯度"""
        if isinstance(net, nn.Module):
            params = [p for p in net.parameters() if p.requires_grad]
        else:
            params = net.params
        norm = torch.sqrt(torch.sum(sum(torch.sum((p.grad**2)) for p in params)))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm

    return


if __name__ == "__main__":
    app.run()
