import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    from brain_computer_interface.convnet.shallow_convnet import ShallowConvNet
    import numpy as np
    import mne
    from loguru import logger
    from braindecode.datasets import  BNCI2014_001

    return BNCI2014_001, ShallowConvNet, np


@app.cell
def _(BNCI2014_001):
    dataset = BNCI2014_001(subject_ids=[1])
    return (dataset,)


@app.cell
def _(dataset):
    dataset
    return


@app.cell
def _(dataset, np):
    from braindecode.preprocessing import (
        Preprocessor,
        exponential_moving_standardize,
        preprocess,
    )

    low_cut_hz = 4.0  # low cut frequency for filtering
    high_cut_hz = 38.0  # high cut frequency for filtering
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor(
            lambda data, factor: np.multiply(data, factor),  # Convert from V to uV
            factor=1e6,
        ),
        Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
        Preprocessor(
            exponential_moving_standardize,  # Exponential moving standardization
            factor_new=factor_new,
            init_block_size=init_block_size,
        ),
    ]

    # Preprocess the data
    preprocess(dataset, preprocessors, n_jobs=-1)

    dataset
    return


@app.cell
def _(dataset):
    from braindecode.preprocessing import create_windows_from_events

    trial_start_offset_seconds = -0.5
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
    )

    windows_dataset
    return (windows_dataset,)


@app.cell
def _(windows_dataset):
    splitted = windows_dataset.split("session")
    train_set = splitted['0train']  # Session train
    test_set = splitted['1test']  # Session evaluation

    splitted,train_set,test_set
    return test_set, train_set


@app.cell
def _(ShallowConvNet, windows_dataset):
    import torch
    from braindecode.util import set_random_seeds

    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = "cuda:1" if cuda else "cpu"
    if cuda:
        torch.backends.cudnn.benchmark = True
    seed = 202601102
    set_random_seeds(seed=seed, cuda=cuda)

    n_classes = 4
    classes = list(range(n_classes))
    # Extract number of chans and time steps from dataset
    n_channels = windows_dataset[0][0].shape[0]
    input_window_samples = windows_dataset[0][0].shape[1]

    model = ShallowConvNet()

    # Display torchinfo table describing the model
    print(model)

    # Send model to GPU
    if cuda:
        model.cuda()
    return classes, device, model, torch


@app.cell
def _(train_set):
    print(len(train_set))
    return


@app.cell
def _(classes, device, model, test_set, torch, train_set):

    from skorch.callbacks import LRScheduler

    from braindecode import EEGClassifier

    lr = 0.0625 * 0.01
    weight_decay = 0
    batch_size = 64
    n_epochs = 2

    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        train_split=None,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=[
            "accuracy",
            ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
        ],
        device=device,
        classes=classes,
        max_epochs=n_epochs,
    )
    # Model training for a specified number of epochs. `y` is None as it is already supplied
    # in the dataset.
    clf.fit(train_set, y=None)

    # evaluated the model after training
    y_test = test_set.get_metadata().target
    test_acc = clf.score(test_set, y=y_test)
    print(f"Test acc: {(test_acc * 100):.2f}%")
    return


if __name__ == "__main__":
    app.run()
