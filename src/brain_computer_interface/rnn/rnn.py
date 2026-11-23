import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import re
    from pathlib import Path
    from loguru import logger

    return Path, logger, re


@app.cell
def _(Path, re):
    def read_time_machine():  #@save
        """将时间机器数据集加载到文本行的列表中"""
        path = Path(__file__).parent / 'timemachine.txt'
        with path.open('r', encoding='utf-8') as f:
            lines = f.readlines()
        return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


    return (read_time_machine,)


@app.cell
def _(logger, read_time_machine):
    lines = read_time_machine()
    # print(f'# 文本总行数: {len(lines)}')
    # print(lines[0])
    # print(lines[10])
    # print(f'# 文本总行数: {len(lines)}')
    logger.info(f"# 文本总行数: {len(lines)}")
    logger.info(lines[10])
    logger.info(lines[0])
    return (lines,)


@app.cell
def _(lines):
    def tokenize(lines, token="word"):  # @save
        """将文本行拆分为单词或字符词元"""
        if token == "word":
            return [line.split() for line in lines]
        elif token == "char":
            return [list(line) for line in lines]
        else:
            print("错误：未知词元类型：" + token)


    tokens = tokenize(lines)
    for i in range(11):
        print(tokens[i])
    return tokenize, tokens


@app.cell
def _():
    import collections


    class Vocab:  # @save
        """文本词表"""

        def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
            if tokens is None:
                tokens = []
            if reserved_tokens is None:
                reserved_tokens = []
            # 按出现频率排序
            counter = count_corpus(tokens)
            self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            # 未知词元的索引为0
            self.idx_to_token = ["<unk>"] + reserved_tokens
            self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
            for token, freq in self._token_freqs:
                if freq < min_freq:
                    break
                if token not in self.token_to_idx:
                    self.idx_to_token.append(token)
                    self.token_to_idx[token] = len(self.idx_to_token) - 1

        def __len__(self):
            return len(self.idx_to_token)

        def __getitem__(self, tokens):
            if not isinstance(tokens, (list, tuple)):
                return self.token_to_idx.get(tokens, self.unk)
            return [self.__getitem__(token) for token in tokens]

        def to_tokens(self, indices):
            if not isinstance(indices, (list, tuple)):
                return self.idx_to_token[indices]
            return [self.idx_to_token[index] for index in indices]

        @property
        def unk(self):  # 未知词元的索引为0
            return 0

        @property
        def token_freqs(self):
            return self._token_freqs


    def count_corpus(tokens):  # @save
        """统计词元的频率"""
        # 这里的tokens是1D列表或2D列表
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # 将词元列表展平成一个列表
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

    return (Vocab,)


@app.cell
def _(Vocab, tokens):
    import matplotlib.pyplot as plt

    vocab = Vocab(tokens)
    print(list(vocab.token_to_idx.items())[:10])
    print(vocab.token_freqs[:10])

    freqs = [freq for token, freq in vocab.token_freqs]

    fig, ax = plt.subplots()
    ax.set_xlabel("token: x")
    ax.set_ylabel("frequency: n(x)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(freqs)
    return freqs, plt, vocab


@app.cell
def _(Vocab, corpus, freqs, plt):
    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = Vocab(bigram_tokens)
    bigram_vocab.token_freqs[:10]

    trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    trigram_vocab = Vocab(trigram_tokens)
    trigram_vocab.token_freqs[:10]

    bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
    trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]

    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("token: x")
    ax1.set_ylabel("frequency: n(x)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    ax1.plot(freqs, linestyle="--", label="1")
    ax1.plot(bigram_freqs, marker="s", linestyle="-", label="2")
    ax1.plot(trigram_freqs, marker="^", linestyle="-.", label="3")
    ax1.legend()
    return


@app.cell
def _(tokens, vocab):
    for j in [0, 10]:
        print("文本:", tokens[j])
        print("索引:", vocab[tokens[j]])
    return


@app.cell
def _(Vocab, read_time_machine, tokenize):
    def load_corpus_time_machine(max_tokens=-1):  # @save
        """返回时光机器数据集的词元索引列表和词表"""
        lines = read_time_machine()
        tokens = tokenize(lines, "char")
        vocab = Vocab(tokens)
        # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
        # 所以将所有文本行展平到一个列表中
        corpus = [vocab[token] for line in tokens for token in line]
        if max_tokens > 0:
            corpus = corpus[:max_tokens]
        return corpus, vocab


    corpus, vocabs = load_corpus_time_machine()
    len(corpus), len(vocabs)
    print(list(vocabs.token_to_idx.items())[:])
    return corpus, load_corpus_time_machine


@app.cell
def _():
    import random
    import torch


    def seq_data_iter_random(corpus, batch_size, num_steps):  # @save
        """使用随机抽样生成一个小批量子序列"""
        # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
        corpus = corpus[random.randint(0, num_steps - 1) :]
        # 减去1，是因为我们需要考虑标签
        num_subseqs = (len(corpus) - 1) // num_steps
        # 长度为num_steps的子序列的起始索引
        initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
        # 在随机抽样的迭代过程中，
        # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
        random.shuffle(initial_indices)

        def data(pos):
            # 返回从pos位置开始的长度为num_steps的序列
            return corpus[pos : pos + num_steps]

        num_batches = num_subseqs // batch_size
        for i in range(0, batch_size * num_batches, batch_size):
            # 在这里，initial_indices包含子序列的随机起始索引
            initial_indices_per_batch = initial_indices[i : i + batch_size]
            X = [data(j) for j in initial_indices_per_batch]
            Y = [data(j + 1) for j in initial_indices_per_batch]
            yield torch.tensor(X), torch.tensor(Y)

    return random, seq_data_iter_random, torch


@app.cell
def _(seq_data_iter_random):
    my_seq = list(range(35))
    for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)
    return (my_seq,)


@app.cell
def _(random, torch):
    def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
        """使用顺序分区生成一个小批量子序列"""
        # 从随机偏移量开始划分序列
        offset = random.randint(0, num_steps)
        num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
        Xs = torch.tensor(corpus[offset: offset + num_tokens])
        Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
        Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
        num_batches = Xs.shape[1] // num_steps
        for i in range(0, num_steps * num_batches, num_steps):
            X = Xs[:, i: i + num_steps]
            Y = Ys[:, i: i + num_steps]
            yield X, Y

    return (seq_data_iter_sequential,)


@app.cell
def _(my_seq, seq_data_iter_sequential):
    for X1, Y1 in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print('X: ', X1, '\nY:', Y1)
    return


@app.cell
def _(
    load_corpus_time_machine,
    seq_data_iter_random,
    seq_data_iter_sequential,
):
    class SeqDataLoader:  # @save
        """加载序列数据的迭代器"""

        def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
            if use_random_iter:
                self.data_iter_fn = seq_data_iter_random
            else:
                self.data_iter_fn = seq_data_iter_sequential
            self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
            self.batch_size, self.num_steps = batch_size, num_steps

        def __iter__(self):
            return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

    return (SeqDataLoader,)


@app.cell
def _(SeqDataLoader):
    def load_data_time_machine(batch_size, num_steps,  #@save
                               use_random_iter=False, max_tokens=10000):
        """返回时光机器数据集的迭代器和词表"""
        data_iter = SeqDataLoader(
            batch_size, num_steps, use_random_iter, max_tokens)
        return data_iter, data_iter.vocab

    return


@app.cell
def _(torch):
    X2, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
    H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
    torch.matmul(X2, W_xh) + torch.matmul(H, W_hh)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
