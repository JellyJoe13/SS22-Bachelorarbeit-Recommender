import json
import typing

import torch


class DataProtocoller:
    def __init__(
            self,
            name: str,
            loss_name: str
    ):
        self.name = name
        self.__epoch_dict = {
            "name": name,
            "loss_name": loss_name
        }

    def register_loss(
            self,
            epoch: int,
            loss: torch.Tensor
    ) -> None:
        if epoch not in self.__epoch_dict[epoch]:
            self.__epoch_dict[epoch] = {}
        self.__epoch_dict[epoch]["loss"] = float(loss)
        return

    def register_roc_auc(
            self,
            epoch: int,
            roc_auc: float
    ) -> None:
        if epoch not in self.__epoch_dict[epoch]:
            self.__epoch_dict[epoch] = {}
        self.__epoch_dict[epoch]["roc_auc"] = roc_auc
        return

    def register_precision_recall(
            self,
            epoch: int,
            precision: typing.Union[float, typing.Tuple[float, float]],
            recall: typing.Union[float, typing.Tuple[float, float]]
    ) -> None:
        if epoch not in self.__epoch_dict[epoch]:
            self.__epoch_dict[epoch] = {}
        if type(precision) == tuple:
            self.__epoch_dict[epoch]["precision"] = {
                "constant": precision[0],
                "relative": precision[1]
            }
        else:
            self.__epoch_dict[epoch]["precision"] = precision
        if type(recall) == tuple:
            self.__epoch_dict[epoch]["recall"] = {
                "constant": recall[0],
                "relative": recall[1]
            }
        else:
            self.__epoch_dict[epoch]["recall"] = recall
        return

    def print_to_console(
            self
    ) -> None:
        print(self.__epoch_dict)
        return

    def save_to_file(
            self,
            file_path: str
    ) -> None:
        if file_path[len(file_path) - 1] != '/':
            file_path = file_path + '/'
        with open(file_path + self.name + ".json", 'w') as fp:
            json.dump(self.__epoch_dict, fp, indent=4)
