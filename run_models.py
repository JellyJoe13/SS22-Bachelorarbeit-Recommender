# todo: write whole execution scheme of epochs.... centralized for all models - partly finished II
# todo: add validation data - control necessary
# todo: early stopping - control necessary
# todo: measure overfitting with steffen things
# todo: setup models

# import section
from typing import Union
import torch
import edge_batch
import run_tools
import utils.data_protocoller
from model_workspace.GNN_gcnconv_testspace import GNN_GCNConv_homogen
from datetime import datetime
from utils.model_config import ModelLoader
from utils.stopping_control import EarlyStoppingControl


def run_epoch(
        model: Union[GNN_GCNConv_homogen],
        optimizer: torch.optim.Optimizer,
        train_batcher: edge_batch.EdgeConvolutionBatcher,
        test_batcher: edge_batch.EdgeConvolutionBatcher,
        model_id: int,
        device
):
    # todo: do separation in model_loader?
    if model_id in [0, 1, 2]:
        loss = run_tools.train_model(model, train_batcher, optimizer)
        roc_auc = run_tools.test_model_basic(model, test_batcher, device)
        return loss, roc_auc
    else:
        return None


def full_test_run(
        model: Union[GNN_GCNConv_homogen],
        batcher: edge_batch.EdgeConvolutionBatcher,
        model_id: int,
        epoch: int,
        split_mode: int,
        device
):
    if model_id in [0, 1, 2]:
        # todo: do separation in model_loader?
        precision, recall = run_tools.test_model_advanced(model, batcher, model_id, device, epoch=epoch,
                                                          split_mode=split_mode)
        return precision, recall
    else:
        return None


def full_experimental_run(
        max_epochs: int,
        model_loader: ModelLoader,
        model_id: int,
        split_mode: int = 0
):
    # load model to test
    model = model_loader.load_model(model_id)
    # load data
    batchers = model_loader.load_model_data(model_id, do_val_split=True, split_mode=split_mode)
    # initialize data and optimizers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters())
    # todo: control device passing controlling
    # initialize data protocoller
    name = "Split-" + str(split_mode) + "/" + model.get_name() + "-" + str(model_id) + "_" \
           + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    protocoller = utils.data_protocoller.DataProtocoller(name)
    # initialize early stopping controller
    esc = EarlyStoppingControl()
    # run epochs
    for epoch in range(max_epochs):
        loss, roc_auc = run_epoch(model, optimizer, batchers["train"], batchers["test"], model_id, device)
        protocoller.register_loss(epoch, loss)
        protocoller.register_roc_auc(epoch, roc_auc)
        if model_loader.should_execute_esc(model_id):
            val_roc_auc = run_tools.test_model_basic(model, batchers["val"], device)
            # pass data to early stopping control
            esc.put_in_roc_auc(val_roc_auc)
            if esc.get_should_stop():
                break
        if epoch % 5 == 0:
            precision, recall = full_test_run(model, batchers["train"], model_id, epoch, split_mode, device)
            protocoller.register_precision_recall(epoch, precision, recall)
        # todo: print data to command line to get progress
    protocoller.save_to_file("experiment_data")
