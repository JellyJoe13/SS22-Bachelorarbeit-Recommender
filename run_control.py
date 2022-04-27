import torch
import torch_geometric.data

import run_tools
from model_config import ModelLoader
from utils.data_related.data_protocoller import DataProtocoller
from datetime import datetime
import model_workspace.GNN_fullbatch_homogen_GCNConv


class RunControl:
    def __init__(
            self,
            model_id: int,
            split_mode: int = 0,
            batch_size: int = 1000000
    ):
        # set basic parameters
        self.protocol_dict = {}
        self.model_id = model_id
        self.batch_size = batch_size
        self.model_loader = ModelLoader()
        # initialize training necessities
        if self.model_loader.works_on_cuda(model_id):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.optimizer = torch.optim.Adam(params=self.model.parameters())
        # load data
        self.data_object = self.__load_data(self.model_loader, split_mode, model_id, batch_size)
        # load model and transfer it to device
        self.model = self.model_loader.load_model(model_id)
        # set parameters for trainings iteration
        self.current_train_epoch = 0
        self.next_train_iteration = 0
        self.iteration_len = len(self.data_object["train"])

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
        name = "Split-" + str(split_mode) + "/" + model_name + "-" + str(model_id) + "_" \
               + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # initialize protocoller
        self.protocoller = DataProtocoller(name, loss_name)

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
            for index in range(self.next_train_iteration, end_iteration):
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
                self.protocoller.register_loss(self.current_train_epoch, self.protocol_dict)
                self.protocol_dict = {}
                self.current_train_epoch += 1
                self.next_train_iteration = 0
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
            self.protocoller.register_loss(self.current_train_epoch, loss)
            self.current_train_epoch += 1
        return

    def do_val_test(
            self
    ):
        if self.model_loader.is_batched():
            None
            # unfinished
