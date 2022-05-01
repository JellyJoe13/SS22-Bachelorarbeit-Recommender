import typing

import torch
import torch_geometric.data

import run_tools
import utils.data_related.edge_batch
from model_config import ModelLoader
from utils.data_related.data_protocoller import DataProtocoller
from datetime import datetime
import model_workspace.GNN_fullbatch_homogen_GCNConv

from tqdm.auto import tqdm


class RunControl:
    def __init__(
            self,
            model_id: int,
            split_mode: int = 0,
            batch_size: int = 1000000,
            loaded_data: typing.Union[typing.Dict[str, utils.data_related.edge_batch.EdgeBatcher],
                                      torch_geometric.data.Data] = None
    ):
        # set basic parameters
        self.protocol_dict = {}
        self.model_id = model_id
        self.batch_size = batch_size
        self.split_mode = split_mode
        self.model_loader = ModelLoader()
        # load model and transfer it to device
        self.model = self.model_loader.load_model(model_id)
        # initialize training necessities
        if self.model_loader.works_on_cuda(model_id):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.optimizer = torch.optim.Adam(params=self.model.parameters())
        # load data
        if loaded_data:
            self.data_object = loaded_data
        else:
            self.data_object = self.__load_data(self.model_loader, split_mode, model_id, batch_size)
        # set parameters for trainings iteration
        self.current_train_epoch = 0
        self.next_train_iteration = 0
        self.iteration_len = len(self.data_object["train"])
        # initialize loss function and protocoller
        self.__initialize_loss_and_protocoller(
            self.split_mode,
            self.model.get_name(),
            self.model_id
        )

    @staticmethod
    def __load_data(
            model_loader: ModelLoader,
            split_mode: int,
            model_id: int,
            batch_size: int
    ):
        return model_loader.load_model_data(
            model_id=model_id,
            do_val_split=True,
            split_mode=split_mode,
            num_selection_edges_batching=batch_size
        )

    def __initialize_loss_and_protocoller(
            self,
            split_mode: int,
            model_name: str,
            model_id: int
    ):
        # fetch loss
        loss_name, loss_function = self.model_loader.get_loss_function(self.model_id)
        self.loss_function = loss_function
        # define logic for naming
        name = "Split-" + str(split_mode) + "_" + model_name + "-" + str(model_id) + "_" \
               + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # initialize protocoller
        self.protocoller = DataProtocoller(name, loss_name, model_id)

    def do_train_step(
            self,
            num_step: int
    ):
        if self.model_loader.is_batched(self.model_id):
            # determine length for for loop
            end_iteration = self.next_train_iteration + num_step
            if end_iteration > len(self.data_object["train"]):
                end_iteration = len(self.data_object["train"])
            # batched model
            for index in tqdm(range(self.next_train_iteration, end_iteration)):
                batch = self.data_object["train"](index).to(self.device)
                loss, roc_auc = run_tools.train_model_batch(
                    model=self.model,
                    optimizer=self.optimizer,
                    data=batch,
                    loss_function=self.loss_function
                )
                batch = batch.detach()
                # put collected information to protocol dict
                self.protocol_dict[index] = {
                    "loss": float(loss),
                    "roc_auc": roc_auc
                }
            # for loop ended
            # in case we reached max_iteration for this epoch write epoch information to protocoller
            if end_iteration == len(self.data_object["train"]):
                self.protocoller.register_train_data(self.current_train_epoch, self.protocol_dict)
                self.protocol_dict = {}
                self.current_train_epoch += 1
                self.next_train_iteration = 0
            else:
                self.next_train_iteration += num_step
        else:
            # fullbatch model
            assert type(self.data_object) == torch_geometric.data.Data
            loss = model_workspace.GNN_fullbatch_homogen_GCNConv.train(
                model=self.model,
                optimizer=self.optimizer,
                data=self.data_object,
                loss_function=self.loss_function
            )
            # todo: soll vielleicht auch roc von train gleichzeitig messen
            self.protocoller.register_train_data(self.current_train_epoch, loss)
            self.current_train_epoch += 1
        return

    def do_val_test(
            self
    ):
        if self.model_loader.is_batched(self.model_id):
            val_batcher = self.data_object["val"]
            loss, roc_auc = run_tools.test_model_basic(
                model=self.model,
                batcher=val_batcher,
                device=self.device,
                loss_function=self.loss_function
            )
            self.protocoller.register_val_data(
                self.current_train_epoch,
                loss,
                roc_auc,
                self.next_train_iteration
            )
        else:
            loss, roc_auc = model_workspace.GNN_fullbatch_homogen_GCNConv.test_with_loss(
                model=self.model,
                data=self.data_object,
                learn_model="val",
                loss_function=self.loss_function
            )
            self.protocoller.register_val_data(
                epoch=self.current_train_epoch,
                roc_auc=roc_auc,
                loss=loss
            )
        return

    def do_test_test(
            self
    ):
        if self.model_loader.is_batched(self.model_id):
            # model is batched
            test_batcher = self.data_object["test"]
            loss, roc_auc = run_tools.test_model_basic(
                model=self.model,
                batcher=test_batcher,
                device=self.device,
                loss_function=self.loss_function
            )
            self.protocoller.register_test_data(
                epoch=self.current_train_epoch,
                iteration=self.next_train_iteration,
                roc_auc=roc_auc,
                loss=loss
            )
        else:
            loss, roc_auc = model_workspace.GNN_fullbatch_homogen_GCNConv.test_with_loss(
                model=self.model,
                data=self.data_object,
                learn_model="test",
                loss_function=self.loss_function
            )
            self.protocoller.register_val_data(
                epoch=self.current_train_epoch,
                roc_auc=roc_auc,
                loss=loss
            )
        return

    def do_full_test(self):
        if self.model_loader.is_batched(self.model_id):
            precision, recall = run_tools.test_model_advanced(
                model=self.model,
                batcher=self.data_object["test"],
                model_id=self.model_id,
                device=self.device,
                epoch=self.current_train_epoch,
                split_mode=self.split_mode
            )
            self.protocoller.register_precision_recall(self.current_train_epoch, precision, recall)
        else:
            precision, recall = model_workspace.GNN_fullbatch_homogen_GCNConv.full_test(
                model=self.model,
                data=self.data_object,
                model_id=self.model_id,
                epoch=self.current_train_epoch,
                split_mode=self.split_mode
            )
            self.protocoller.register_precision_recall(precision, recall)
        return

    def run_epoch(
            self,
            val_test_frequency: int
    ):
        if self.model_loader.is_batched(self.model_id):
            assert val_test_frequency > 0
            # for loop over train batcher
            for iteration in tqdm(range(0, len(self.data_object["train"]), val_test_frequency)):
                self.do_train_step(val_test_frequency)
                self.do_val_test()
                self.do_test_test()
        else:
            self.do_train_step(val_test_frequency)
            self.do_val_test()
            self.do_test_test()
        self.do_full_test()
        return

    # todo: use early stopping or maybe just for final predicting program
    def run_experiment(
            self,
            val_test_frequency: int,
            max_epochs: int
    ):
        self.do_full_test()
        self.do_val_test()
        self.do_test_test()
        for epoch in tqdm(range(max_epochs)):
            self.run_epoch(val_test_frequency)
        return

    def save_protocoll(self):
        self.protocoller.save_to_file("experimental_results")
