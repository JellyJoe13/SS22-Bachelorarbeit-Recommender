import json
import typing

import torch


class DataProtocoller:
    """
    Data Protocoller used for centrally collecting the accuracy scores, losses and adta in general that is produced
    while training the model. Can print or save data.
    """
    def __init__(
            self,
            name: str,
            loss_name: str,
            model_id: int
    ):
        """
        Initialize the DataProtocoller class with the name of the model or the experiment and the used loss name which
        will be written in the data dictionary.

        Parameters
        ----------
        model_id : int
            Denotes model identifier with which the model can be polled from model_config.py
        name : str
            specifies the name of the model or experiment
        loss_name : str
            specifies the name of the used loss while executing the model
        """
        self.name = name
        self.__epoch_dict = {
            "name": name,
            "loss_name": loss_name,
            "model_id": model_id,
            "training": {},
            "validating": {},
            "testing": {},
            "accuracy": {}
        }

    def register_train_data(
            self,
            epoch: int,
            train_protocol: typing.Union[float,
                                         typing.Dict[str, float],
                                         typing.Dict[int, typing.Dict[str, float]]]
    ) -> None:
        """
        Function that allows for tracking the loss of the training and allowing it for saving purposes.

        Parameters
        ----------
        epoch : int
            Information on which epoch the loss corresponds to
        train_protocol : torch.Tensor
            Loss which was computed in the training phase of the epoch that was supplied.

        Returns
        -------
        Nothing
        """
        self.__epoch_dict["training"][epoch] = train_protocol
        return

    def register_test_data(
            self,
            epoch: int,
            iteration: int = 0,
            roc_auc: float = None,
            loss: float = None
    ) -> None:
        """
        Function that allows for tracking the roc auc and loss of the testing and allowing it for saving purposes.

        Parameters
        ----------
        loss : float
            loss to be tracked for test data
        iteration : int
            additional information in case the test was not conducted between training epochs but between learn batches.
        epoch : int
            Information on which epoch the roc auc corresponds to
        roc_auc : int
            ROC AUC which was computed in the test phase of the epoch that was supplied

        Returns
        -------
        Nothing
        """
        if epoch not in self.__epoch_dict["testing"]:
            self.__epoch_dict["testing"][epoch] = {}
        self.__epoch_dict["testing"][epoch][iteration] = {
            "loss": loss,
            "roc_auc": roc_auc
        }
        return

    def register_val_data(
            self,
            epoch: int,
            loss: float = None,
            roc_auc: float = None,
            iteration: int = 0
    ) -> None:
        """
        Function that allows for tracking of the validation loss and roc auc.

        Parameters
        ----------
        epoch
        loss
        roc_auc
        iteration

        Returns
        -------
        Nothing
        """
        if epoch not in self.__epoch_dict["validating"]:
            self.__epoch_dict["validating"][epoch] = {}
        val_epoch_entry = self.__epoch_dict["validating"][epoch]
        val_epoch_entry[iteration] = {
            "loss": loss,
            "roc_auc": roc_auc
        }
        return

    def register_precision_recall(
            self,
            epoch: int,
            precision: typing.Union[float, typing.Tuple[float, float]],
            recall: typing.Union[float, typing.Tuple[float, float]]
    ) -> None:
        """
        Function that allows for tracking the precision and recall of the full testing phase and allowing it for saving
        purposes.

        Parameters
        ----------
        epoch : int
            Information on which epoch the roc auc corresponds to
        precision : typing.Union[float, typing.Tuple[float, float]]
            Precision that was supplied for saving and tracking purposes
        recall : typing.Union[float, typing.Tuple[float, float]]
            Recall that was supplied for saving and tracking purposes

        Returns
        -------
        Nothing
        """
        if epoch not in self.__epoch_dict["accuracy"]:
            self.__epoch_dict["accuracy"][epoch] = {}
        if type(precision) == tuple:
            self.__epoch_dict["accuracy"][epoch]["precision"] = {
                "constant": precision[0],
                "relative": precision[1]
            }
        else:
            self.__epoch_dict["accuracy"][epoch]["precision"] = precision
        if type(recall) == tuple:
            self.__epoch_dict["accuracy"][epoch]["recall"] = {
                "constant": recall[0],
                "relative": recall[1]
            }
        else:
            self.__epoch_dict[epoch]["recall"] = recall
        return

    def print_to_console(
            self
    ) -> None:
        """
        Function that prints the saved results to console.

        Returns
        -------
        Nothing
        """
        print(self.__epoch_dict)
        return

    def save_to_file(
            self,
            file_path: str
    ) -> None:
        """
        Function that saves the collected data to a json file.

        Parameters
        ----------
        file_path : str
            Specifies the path to which the file should be saved to. MAY NOT CONTAIN THE ACTUAL FILE NAME and a data
            ending

        Returns
        -------
        Nothing
        """
        if file_path[len(file_path) - 1] != '/':
            file_path = file_path + '/'
        with open(file_path + self.name + ".json", 'w') as fp:
            json.dump(self.__epoch_dict, fp, indent=4)
