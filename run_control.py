import typing

import numpy as np
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
    """
    Class used for detailed execution of model training, testing and validation.

    It is recommended to only use this class for batched models as fullbatch models have proved to be possibly not
    working to a too high memory allocation requirement in contrast to the execution of the fullbatch model in the
    jupyter notebook.
    """

    def __init__(
            self,
            model_id: int,
            split_mode: int = 0,
            batch_size: int = 1000000,
            loaded_data: typing.Union[typing.Dict[str, utils.data_related.edge_batch.EdgeBatcher],
                                      torch_geometric.data.Data] = None,
            test_mode: bool = False
    ):
        """
        Initialize the class including the loading and preparing of data and model.

        Parameters
        ----------
        test_mode : bool
            defines whether the mini-data-set or the large data-set shall be loaded
        model_id : int
            id of the model to be loaded and used for training, testing and validation operations. ids defined in
            model_config.py
        split_mode : int
            split mode for data loading
        batch_size : int
            size of the batch data object a.k.a. the number of edges to predict/train in the batch data objects
        loaded_data : typing.Union[typing.Dict[str, utils.data_related.edge_batch.EdgeBatcher],
                                   torch_geometric.data.Data]
            in case data for this model has already been loaded it can be supplied via this parameter
        """
        # set basic parameters
        self.test_mode = test_mode
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
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters())
        # load data
        if loaded_data:
            self.data_object = loaded_data
        else:
            self.data_object, self.data_info = self.__load_data(
                self.model_loader,
                split_mode,
                model_id,
                batch_size,
                self.test_mode
            )
        # set parameters for trainings iteration
        self.current_train_epoch = 0
        self.next_train_iteration = 0
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
            batch_size: int,
            test_mode: bool = False
    ):
        """
        Method to load the data of the corresponding model.

        Parameters
        ----------
        test_mode : bool
            specifies whether the mini dataset or the full dataset shall be loaded from disk in case of pytorch data
            option
        model_loader : ModelLoader
            model loader which is used to load the model data from
        split_mode : int
            split mode of the data to load
        model_id : int
            id of model to load the data for
        batch_size : int
            size of the batch data object

        Returns
        -------
        loaded data
        """
        return model_loader.load_model_data(
            model_id=model_id,
            do_val_split=True,
            split_mode=split_mode,
            num_selection_edges_batching=batch_size,
            test_mode=test_mode
        )

    def __initialize_loss_and_protocoller(
            self,
            split_mode: int,
            model_name: str,
            model_id: int
    ):
        """
        Initialize loss function and data protocoller.

        Parameters
        ----------
        split_mode : int
            split mode of the data; used for name giving for the protocol file
        model_name : str
            name of the model; used for name giving for the protocol file
        model_id : int
            id of the model; used for name giving for the protocol file

        Returns
        -------
        None
        """
        # fetch loss
        loss_name, loss_function = self.model_loader.get_loss_function(self.model_id)
        self.loss_function = loss_function
        # define logic for naming
        name = "Split-" + str(split_mode) + "_" + model_name + "-" + str(model_id) + "_" \
               + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # initialize protocoller
        self.protocoller = DataProtocoller(name, loss_name, model_id)
        return

    def do_train_step(
            self,
            num_step: int
    ):
        """
        Function used for execute a training step which includes the partial execution of training iterations for an
        epoch with num_step iterations being executed.

        Parameters
        ----------
        num_step : int
            number of training iterations being carried out

        Returns
        -------
        None
        """
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
            loss, roc_auc = model_workspace.GNN_fullbatch_homogen_GCNConv.train_with_roc_auc(
                model=self.model,
                optimizer=self.optimizer,
                data=self.data_object,
                loss_function=self.loss_function
            )
            self.protocoller.register_train_data(
                self.current_train_epoch,
                {
                    "loss": float(loss),
                    "roc_auc": roc_auc
                }
            )
            self.current_train_epoch += 1
        return

    def do_val_test(
            self
    ):
        """
        Function used to carry out a validation operating measuring the loss and roc auc of the validation dataset.

        Returns
        -------
        None
        """
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
        """
        Function used to carry out a testing operating measuring the loss and roc auc of the test dataset.

        Returns
        -------
        None
        """
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
        """
        Function used to carry out a full test operation measuring the accuracy scores and creating a ROC plot that
        will be saved to a svd file by default.

        Returns
        -------
        None
        """
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
            val_test_frequency: int,
            fluctuation_control_mode: int = 0
    ):
        """
        Function used to execute all the necessary operations for a whole epoch.

        Parameters
        ----------
        fluctuation_control_mode : int
            defines if a measure should be taken to prevent loss fluctuation. 0 for no fluctuation control, 1 for
            randomizing between epochs and 2 for leaving last batch (which is potentially smaller) out.
        val_test_frequency : int
            determines after how many train iterations a test on the validation and test set will be carried out.

        Returns
        -------
        None
        """
        # checks if mode in range
        assert 0 <= fluctuation_control_mode <= 2
        # mode where the training set is randomized between epochs
        if fluctuation_control_mode == 1:
            self.data_object['train'].randomize()
        if self.model_loader.is_batched(self.model_id):
            assert val_test_frequency > 0
            # for loop over train batcher
            for iteration in tqdm(range(0, len(self.data_object["train"]), val_test_frequency)):
                # detect end of for loop and not execute last train batch
                if fluctuation_control_mode == 2 \
                        and iteration == ((len(self.data_object['train']) - 1) % val_test_frequency):
                    continue
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
            max_epochs: int,
            fluctuation_control_mode: int = 0
    ):
        """
        Function used to execute a whole experiment for the loaded model, data and settings in general. Also saves the
        tracked data at the end of each epoch to the json output file.

        Parameters
        ----------
        fluctuation_control_mode : int
            defines if a measure should be taken to prevent loss fluctuation. 0 for no fluctuation control, 1 for
            randomizing between epochs and 2 for leaving last batch (which is potentially smaller) out.
        val_test_frequency : int
            determines after how many train iterations a test on the validation and test set will be carried out.
        max_epochs : int
            determines the maximal number of epochs that will be run in the experiment.

        Returns
        -------
        None
        """
        self.do_full_test()
        self.do_val_test()
        self.do_test_test()
        for epoch in tqdm(range(max_epochs)):
            self.run_epoch(val_test_frequency, fluctuation_control_mode=fluctuation_control_mode)
            self.save_protocoll()
        return

    def save_protocoll(self):
        """
        Function that saves the collected data to a json file.

        Returns
        -------
        None
        """
        self.protocoller.save_to_file("experimental_results")
        return

    @torch.no_grad()
    def predict_edges(
            self,
            molecule_ids: np.ndarray,
            experiment_ids: np.ndarray
    ) -> np.ndarray:
        # transform numpy input information on which molecule-experiment pair to predict to torch tensor
        transformed_edges = self.data_info.create_mapped_edges(
            cid=molecule_ids,
            aid=experiment_ids
        )
        # if model is batched (and cuda should be applied) split edges to predict in batch_size sizes
        if self.model_loader.is_batched(self.model_id):
            # shortcut for original data link
            original_data = self.data_object["train"].original_data
            # todo: check for correctness
            # initialize logits
            logits = []
            # iterate over subsets of edges to predict
            for idx_start in range(0, transformed_edges.size(1), self.data_object["train"].num_selection_edges):
                # calculate starting index and end index of edge subset
                idx_end = idx_start + self.data_object["train"].num_selection_edges
                if idx_end > transformed_edges.size(1):
                    idx_end = transformed_edges.size(1)
                # create prediction and append result to list
                logits.append(self.model.fit_predict(
                    original_data.x.to(self.device),
                    transformed_edges[:, idx_start:idx_end].to(self.device),
                    original_data.train_pos_edge_index.to(self.device)
                ).sigmoid().detach().cpu().numpy())
            # fuze and return
            return np.concatenate(logits)
