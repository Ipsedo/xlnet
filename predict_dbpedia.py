import pandas
from run_classifier import DBPedia_filtered
from torchnet.meter import ConfusionMeter
import matplotlib.pyplot as plt
import torch as th

if __name__ == "__main__":
    predict_file = "./exp/dbpedia_filtered/predict/dbpedia_filtered.tsv"
    test_data = DBPedia_filtered().get_dev_examples("")

    df = pandas.read_csv(predict_file, header=0, sep="\t")

    conf_meter = ConfusionMeter(87, normalized=True)
    class_to_idx = {}
    for data, y_pred in zip(test_data, df.iloc[:, 1]):
        if data.label not in class_to_idx:
            class_to_idx[data.label] = len(class_to_idx)
        if y_pred not in class_to_idx:
            class_to_idx[y_pred] = len(class_to_idx)

        conf_meter.add(th.tensor([class_to_idx[data.label]]), th.tensor([class_to_idx[y_pred]]))

    plt.matshow(conf_meter.value())
    plt.colorbar()
    plt.title("Confusion Matrix - XLNet - DBPedia (87 classes)")
    plt.legend()
    plt.show()


