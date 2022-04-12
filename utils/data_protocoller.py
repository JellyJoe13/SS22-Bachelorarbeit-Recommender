import json

import torch


class DataProtocoller:
    def __init__(
            self,
            name: str
    ):
        self.name = name
        self.__epoch_dict = {}

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
            precision: float,
            recall: float
    ) -> None:
        if epoch not in self.__epoch_dict[epoch]:
            self.__epoch_dict[epoch] = {}
        self.__epoch_dict[epoch]["precision"] = precision
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
