import json
import matplotlib.pyplot as plt


class DataInterpreter:
    def __init__(
            self,
            path_to_file: str,
            batched: bool = True
    ):
        self.batched = batched
        with open(path_to_file, 'r') as f:
            load_dict = json.load(f)
        assert load_dict
        self.data_store = self.check_and_correct_loaded_dict(load_dict, batched)

    @staticmethod
    def check_and_correct_loaded_dict(
            load_dict: dict,
            batched: bool
    ):
        # training, validating and testing section
        for mode in ["training", "validating", "testing"]:
            assert mode in load_dict
            load_dict[mode] = dict(zip([int(i) for i in load_dict[mode].keys()],
                                       load_dict[mode].values()))
            if batched:
                for i in load_dict[mode].keys():
                    load_dict[mode][i] = dict(zip([int(i) for i in load_dict[mode][i].keys()],
                                                  load_dict[mode][i].values()))
        # accuracy section
        assert "accuracy" in load_dict
        load_dict["accuracy"] = dict(zip([int(i) for i in load_dict["accuracy"].keys()],
                                         load_dict["accuracy"].values()))
        return load_dict

    def create_loss_plot(self):
        epochs = list(set(list(self.data_store["training"].keys())
                          + list(self.data_store["validating"].keys())
                          + list(self.data_store["testing"].keys())))
        epoch_mapping = [0]
        if self.batched:
            train_loss = {}
            val_loss = {}
            test_loss = {}
            for epoch in epochs:
                epoch_it_size = len(self.data_store["training"][epoch]) if epoch in self.data_store["training"] else 1
                epoch_mapping.append(epoch_mapping[epoch] + epoch_it_size)
                # training
                if epoch not in self.data_store["training"]:
                    continue
                for key, value in self.data_store["training"][epoch].items():
                    train_loss[key + epoch_mapping[epoch]] = value
                # validating
                if epoch not in self.data_store["validating"]:
                    continue
                for key, value in self.data_store["validating"][epoch].items():
                    val_loss[key + epoch_mapping[epoch]] = value
                # testing
                if epoch not in self.data_store["testing"]:
                    continue
                for key, value in self.data_store["testing"][epoch].items():
                    test_loss[key + epoch_mapping[epoch]] = value
            # rework so that it only contains loss
            train_loss = dict(zip(train_loss.keys(), [i["loss"] for i in train_loss.values()]))
            val_loss = dict(zip(val_loss.keys(), [i["loss"] for i in val_loss.values()]))
            test_loss = dict(zip(test_loss.keys(), [i["loss"] for i in test_loss.values()]))
            # todo: plot this
            return epoch_mapping, train_loss, val_loss, test_loss
        else:
            # fullbatched and boring
            # todo: implement this
            None
        # collect data
        # todo: other diagrams
