import json
import matplotlib.pyplot as plt


class DataInterpreter:
    """
    Class to interpret the Data of RunController from run_control.py which was stored in json. Is capable to create
    loss, roc and accuracy score plots.
    """
    def __init__(
            self,
            path_to_file: str,
            batched: bool = True
    ):
        """
        Init function which initializes the DataInterpreter class. Loads the data from json file that was reference with
        the parameter path_to_file.

        Parameters
        ----------
        path_to_file : str
            Specifies from where to load the json data.
        batched : bool
            Specifies if the data is from a batched model or not.
        """
        self.batched = batched
        # load data from file
        with open(path_to_file, 'r') as f:
            load_dict = json.load(f)
        assert load_dict
        # correct loaded data meaning that the int keys are fixed and some structural parts checked
        self.data_store = self.check_and_correct_loaded_dict(load_dict, batched)

    @staticmethod
    def check_and_correct_loaded_dict(
            load_dict: dict,
            batched: bool
    ) -> dict:
        """
        Function used to correct the provided data dict meaning that stringified int values will be returned to int
        format. Some structural checks will be done as well.

        Parameters
        ----------
        load_dict : dict
            loaded data to convert and check
        batched : bool
            Information on whether the model corresponding to the data was batched or not. (different data structure)

        Returns
        -------
        corrected_data : dict
            contains the reformatted dict
        """
        # training, validating and testing section
        for mode in ["training", "validating", "testing"]:
            assert mode in load_dict
            # load transform data
            load_dict[mode] = dict(zip([int(i) for i in load_dict[mode].keys()],
                                       load_dict[mode].values()))
            # if it is batched also convert the iteration keys
            if batched:
                for i in load_dict[mode].keys():
                    load_dict[mode][i] = dict(zip([int(i) for i in load_dict[mode][i].keys()],
                                                  load_dict[mode][i].values()))
        # accuracy transformation section
        assert "accuracy" in load_dict
        load_dict["accuracy"] = dict(zip([int(i) for i in load_dict["accuracy"].keys()],
                                         load_dict["accuracy"].values()))
        return load_dict

    def create_loss_plot(self) -> None:
        """
        Create a diagram for the collected loss values.

        Returns
        -------
        None
        """
        epochs = list(set(list(self.data_store["training"].keys())
                          + list(self.data_store["validating"].keys())
                          + list(self.data_store["testing"].keys())))
        print(epochs)
        epoch_mapping = [0]
        if self.batched:
            train_loss = {}
            val_loss = {}
            test_loss = {}
            for epoch in epochs:
                epoch_it_size = len(self.data_store["training"][epoch]) if epoch in self.data_store["training"] else 1
                epoch_mapping.append(epoch_mapping[epoch] + epoch_it_size)
                # training
                if epoch in self.data_store["training"]:
                    for key, value in self.data_store["training"][epoch].items():
                        train_loss[key + epoch_mapping[epoch]] = value["loss"]
                # validating
                if epoch in self.data_store["validating"]:
                    for key, value in self.data_store["validating"][epoch].items():
                        val_loss[key + epoch_mapping[epoch]] = value["loss"]
                # testing
                if epoch in self.data_store["testing"]:
                    for key, value in self.data_store["testing"][epoch].items():
                        test_loss[key + epoch_mapping[epoch]] = value["loss"]
            # plotting graph
            fig = plt.figure(0)
            plt.plot(train_loss.keys(), train_loss.values(), 'x-', label='train')
            plt.plot(test_loss.keys(), test_loss.values(), 'x-', label='test')
            plt.plot(val_loss.keys(), val_loss.values(), 'x-', label="val")
            plt.title("Tracked losses")
            plt.xlabel("total iterations")
            plt.ylabel("loss value")
            plt.legend()
            for epoch in epoch_mapping:
                plt.axvline(x=epoch, label="epochs", linestyle=':', color='lightgray')
            plt.show()
            return
        else:
            # fullbatched and boring
            train_loss = dict(zip(self.data_store["training"].keys(),
                                  [i["loss"] for i in self.data_store["training"].values()]))
            val_loss = dict(zip(self.data_store["validating"].keys(),
                                [i["loss"] for i in self.data_store["validating"].values()]))
            test_loss = dict(zip(self.data_store["testing"].keys(),
                                 [i["loss"] for i in self.data_store["testing"].values()]))
            fig = plt.figure(0)
            plt.plot(train_loss.keys(), train_loss.values(), 'x-', label='train')
            plt.plot(test_loss.keys(), test_loss.values(), 'x-', label='test')
            plt.plot(val_loss.keys(), val_loss.values(), 'x-', label="val")
            plt.title("Tracked losses")
            plt.xlabel("epochs")
            plt.ylabel("loss value")
            plt.legend()
            plt.show()
            return

    def create_roc_auc_plot(self) -> None:
        """
        Create a diagram for the collected roc auc scores.

        Returns
        -------
        None
        """
        epochs = list(set(list(self.data_store["training"].keys())
                          + list(self.data_store["validating"].keys())
                          + list(self.data_store["testing"].keys())))
        print(epochs)
        epoch_mapping = [0]
        if self.batched:
            train_loss = {}
            val_loss = {}
            test_loss = {}
            for epoch in epochs:
                epoch_it_size = len(self.data_store["training"][epoch]) if epoch in self.data_store["training"] else 1
                epoch_mapping.append(epoch_mapping[epoch] + epoch_it_size)
                # training
                if epoch in self.data_store["training"]:
                    for key, value in self.data_store["training"][epoch].items():
                        train_loss[key + epoch_mapping[epoch]] = value["roc_auc"]
                # validating
                if epoch in self.data_store["validating"]:
                    for key, value in self.data_store["validating"][epoch].items():
                        val_loss[key + epoch_mapping[epoch]] = value["roc_auc"]
                # testing
                if epoch in self.data_store["testing"]:
                    for key, value in self.data_store["testing"][epoch].items():
                        test_loss[key + epoch_mapping[epoch]] = value["roc_auc"]
            # plotting graph
            fig = plt.figure(0)
            plt.plot(train_loss.keys(), train_loss.values(), 'x-', label='train')
            plt.plot(test_loss.keys(), test_loss.values(), 'x-', label='test')
            plt.plot(val_loss.keys(), val_loss.values(), 'x-', label="val")
            plt.title("Tracked ROC AUC scores")
            plt.xlabel("total iterations")
            plt.ylabel("score")
            plt.legend()
            for epoch in epoch_mapping:
                plt.axvline(x=epoch, label="epochs", linestyle=':', color='lightgray')
            plt.show()
            return
        else:
            # fullbatched and boring
            train_loss = dict(zip(self.data_store["training"].keys(),
                                  [i["roc_auc"] for i in self.data_store["training"].values()]))
            val_loss = dict(zip(self.data_store["validating"].keys(),
                                [i["0"]["roc_auc"] for i in self.data_store["validating"].values()]))
            test_loss = dict(zip(self.data_store["testing"].keys(),
                                 [i["roc_auc"] for i in self.data_store["testing"].values()]))
            fig = plt.figure(0)
            plt.plot(train_loss.keys(), train_loss.values(), 'x-', label='train')
            plt.plot(test_loss.keys(), test_loss.values(), 'x-', label='test')
            plt.plot(val_loss.keys(), val_loss.values(), 'x-', label="val")
            plt.title("Tracked ROC AUC scores")
            plt.xlabel("epochs")
            plt.ylabel("score")
            plt.legend()
            plt.show()
            return

    def create_accuracy_plot(self) -> None:
        """
        Create a diagram for the collected accuracy scores.

        Returns
        -------
        None
        """
        precision = dict(zip(self.data_store["accuracy"].keys(),
                             [i["precision"] for i in self.data_store["accuracy"].values()]))
        recall = dict(zip(self.data_store["accuracy"].keys(),
                          [i["recall"] for i in self.data_store["accuracy"].values()]))
        fig = plt.figure(0)
        plt.plot(precision.keys(), [i["constant"] for i in precision.values()], 'x-', label='precision_constant')
        plt.plot(precision.keys(), [i["relative"] for i in precision.values()], 'x-', label='precision_relative')
        plt.plot(recall.keys(), [i["constant"] for i in recall.values()], 'x-', label='recall_constant')
        plt.plot(recall.keys(), [i["relative"] for i in recall.values()], 'x-', label='recall_relative')
        plt.legend()
        plt.title("Accuracy scores")
        plt.xlabel("epochs")
        plt.ylabel("score")
        plt.show()
        return
