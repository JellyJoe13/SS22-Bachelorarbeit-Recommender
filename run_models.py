# todo: write whole execution scheme of epochs.... centralized for all models - partly finished II
# todo: add validation data_related - control necessary
# todo: early stopping - control necessary
# todo: measure overfitting with steffen things
# todo: setup models
# todo: check surpriselib working

# import section
import typing
from typing import Union
import torch
import torch_geometric.data

from utils.data_related import edge_batch
import run_tools
import utils.data_related.data_protocoller
import model_workspace
from datetime import datetime
from model_config import ModelLoader
from utils.accuracy.stopping_control import EarlyStoppingControl
from utils.accuracy import accuracy_surpriselib
from model_workspace.GNN_fullbatch_homogen_GCNConv import GNN_homogen_chemData_GCN
from model_workspace.GNN_minibatch_homogen_GATConv import GNN_GATConv_homogen
from model_workspace.GNN_minibatch_homogen_GCNConv_one import GNN_GCNConv_homogen_basic
from model_workspace.GNN_minibatch_homogen_GCNConv_two import GNN_GCNConv_homogen
from model_workspace.GNN_minibatch_homogen_LGConv_k import GNN_LGConv_homogen_variable
from model_workspace.GNN_minibatch_homogen_SAGEConv import GNN_SAGEConv_homogen


def run_epoch(
        model: Union[GNN_GCNConv_homogen,
                     GNN_homogen_chemData_GCN,
                     GNN_GATConv_homogen,
                     GNN_GCNConv_homogen_basic,
                     GNN_LGConv_homogen_variable,
                     GNN_SAGEConv_homogen],
        optimizer: torch.optim.Optimizer,
        data_object: Union[dict, torch_geometric.data.Data],
        model_id: int,
        device,
        model_loader: ModelLoader,
        loss_function: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
):
    if model_loader.is_batched(model_id):
        assert type(data_object) == dict
        train_batcher = data_object["train"]
        test_batcher = data_object["test"]
        loss = run_tools.train_model(model, train_batcher, optimizer, loss_function=loss_function)
        roc_auc = run_tools.test_model_basic(model, test_batcher, device)
    else:
        assert type(data_object) == torch_geometric.data.Data
        # fullbatch on pytorch
        loss = model_workspace.GNN_fullbatch_homogen_GCNConv.train(model,
                                                                   optimizer,
                                                                   data_object,
                                                                   loss_function=loss_function).detach()
        roc_auc = model_workspace.GNN_fullbatch_homogen_GCNConv.test(model,
                                                                     data_object)
    return loss, roc_auc


def full_test_run(
        model: Union[GNN_GCNConv_homogen,
                     GNN_homogen_chemData_GCN,
                     GNN_GATConv_homogen,
                     GNN_GCNConv_homogen_basic,
                     GNN_LGConv_homogen_variable,
                     GNN_SAGEConv_homogen],
        data_object: Union[edge_batch.EdgeConvolutionBatcher, torch_geometric.data.Data],
        model_id: int,
        epoch: int,
        split_mode: int,
        device,
        model_loader: ModelLoader
):
    # determine if this is a fullbatch or minibatch model
    if model_loader.is_batched():
        assert type(data_object) == edge_batch.EdgeConvolutionBatcher
        precision, recall = run_tools.test_model_advanced(model, data_object, model_id, device, epoch=epoch,
                                                          split_mode=split_mode)
    else:
        assert type(data_object) == torch_geometric.data.Data
        precision, recall = model_workspace.GNN_fullbatch_homogen_GCNConv.full_test(model,
                                                                                    data_object,
                                                                                    model_id,
                                                                                    epoch,
                                                                                    split_mode)
    return precision, recall


# todo: progress indication?
def full_experimental_run(
        model_loader: ModelLoader,
        model_id: int,
        split_mode: int = 0,
        max_epochs: int = 100
):
    # differ between surpriselib and pytorch_geometric models
    if model_loader.is_pytorch(model_id):
        # we have a pytorch model
        # load model to test
        model = model_loader.load_model(model_id)
        # load data_related
        data_object = model_loader.load_model_data(model_id, do_val_split=True, split_mode=split_mode)
        # initialize data_related and optimizers
        device = torch.device('cuda' if (torch.cuda.is_available() and model_loader.works_on_cuda(model_id)) else 'cpu')
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters())
        # todo: control device passing controlling
        # fetch loss and loss name
        loss_name, loss_function = model_loader.get_loss_function(model_id)
        # initialize data_related protocoller
        name = "Split-" + str(split_mode) + "/" + model.get_name() + "-" + str(model_id) + "_" \
               + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        protocoller = utils.data_related.data_protocoller.DataProtocoller(name, loss_name)
        # initialize early stopping controller
        esc = EarlyStoppingControl()
        # run epochs
        for epoch in range(max_epochs):
            loss, roc_auc = run_epoch(model, optimizer, data_object, model_id, device, model_loader, loss_function)
            protocoller.register_loss(epoch, loss)
            protocoller.register_roc_auc(epoch, roc_auc)
            print("Epoch:", epoch, "| loss:", float(loss), "| ROC AUC:", float(roc_auc))
            if model_loader.should_execute_esc(model_id):
                # differ validation_method if we have minibatching enabled
                if model_loader.is_batched(model_id):
                    assert type(data_object) == dict
                    val_roc_auc = run_tools.test_model_basic(model, data_object["val"], device)
                else:
                    assert type(data_object) == torch_geometric.data.Data
                    # we have fullbatch in pytorch
                    val_roc_auc = model_workspace.GNN_fullbatch_homogen_GCNConv.test(model, data_object, "val")
                # pass data_related to early stopping control
                esc.put_in_roc_auc(val_roc_auc)
                # print Information to command line
                print(" - val ROC AUC:", float(val_roc_auc))
                if esc.get_should_stop():
                    break
            if epoch % 5 == 0:
                precision, recall = full_test_run(model, data_object["test"], model_id, epoch, split_mode, device,
                                                  model_loader)
                protocoller.register_precision_recall(epoch, precision, recall)
                # print precision and recall of full_test
                print(" - full_test")
                print("   + precision:", float(precision))
                print("   + recall:", float(recall))
        # execute final full_test
        precision, recall = full_test_run(model, data_object["test"], model_id, -1, split_mode, device, model_loader)
        protocoller.register_precision_recall(-1, precision, recall)
        # save protocolled values to file in folder experiment_data
        protocoller.save_to_file("experiment_data")
        # detach model
        model.detach_()
        return
    else:
        # it is an surpriselib model
        model = model_loader.load_model(model_id)
        trainset, testset = model_loader.load_model_data(model_id, split_mode=split_mode)
        model.fit(trainset)
        predictions = model.test(testset)
        precision, recall = accuracy_surpriselib.accuracy_precision_recall(predictions)
        print("Precision:", float(precision))
        print("Recall:", recall)
        accuracy_surpriselib.calc_ROC_curve(predictions, save_as_file=True, output_file_name="ROC_baseline")
        return
