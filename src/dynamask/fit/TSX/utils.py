import os
import pickle as pkl
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.utils.data as utils
from sklearn.metrics import classification_report, precision_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

from dynamask.fit.TSX.models import PatientData

# np.set_printoptions(threshold=sys.maxsize)
# sns.set()

line_styles_map = [
    "-",
    "--",
    "-.",
    ":",
    "-",
    "--",
    "-.",
    ":",
    "-",
    "--",
    "-.",
    ":",
    "-",
    "--",
    "-.",
    ":",
]
marker_styles_map = [
    "o",
    "v",
    "^",
    "*",
    "+",
    "p",
    "8",
    "h",
    "o",
    "v",
    "^",
    "*",
    "+",
    "p",
    "8",
    "h",
    "o",
    "v",
    "^",
    "*",
    "+",
    "p",
    "8",
    "h",
]

# Ignore sklearn warnings caused by ill-defined precision score (caused by single class prediction)
warnings.filterwarnings("ignore")

intervention_list = [
    "vent",
    "vaso",
    "adenosine",
    "dobutamine",
    "dopamine",
    "epinephrine",
    "isuprel",
    "milrinone",
    "norepinephrine",
    "phenylephrine",
    "vasopressin",
    "colloid_bolus",
    "crystalloid_bolus",
    "nivdurations",
]


def evaluate_binary(labels, predicted_label, predicted_probability):
    labels_array = np.array(labels.cpu())
    prediction_array = np.array(predicted_label.cpu())
    if len(np.unique(labels_array)) < 2:
        auc = 0
    else:
        auc = roc_auc_score(
            np.array(labels.cpu()),
            np.array(predicted_probability.view(len(labels), -1).detach().cpu()),
        )
    recall = torch.matmul(labels, predicted_label).item()
    precision = precision_score(labels_array, prediction_array)
    correct_label = torch.eq(labels, predicted_label).sum()
    return auc, recall, precision, correct_label


def evaluate(labels, predicted_label, predicted_probability):
    labels_array = labels.detach().cpu().numpy()
    prediction_array = predicted_label.detach().cpu().numpy()

    # print(labels_array.shape, predicted_label.shape, predicted_probability.shape)
    if len(np.unique(labels_array)) >= 2:
        auc = roc_auc_score(
            labels_array[:, 1], np.array(predicted_probability[:, 1].detach().cpu())
        )
        report = classification_report(
            labels_array[:, 1], prediction_array[:, 1], output_dict=True
        )
        recall = report["macro avg"]["recall"]
        precision = report["macro avg"]["precision"]
    else:
        auc = 0
        recall = 0
        precision = 0
    correct_label = np.equal(
        np.argmax(labels_array, 1), np.argmax(prediction_array, 1)
    ).sum()
    return auc, recall, precision, correct_label


def evaluate_multiclass(
    labels, predicted_label, predicted_probability, task="multiclass"
):
    labels_array = labels.detach().cpu().numpy()  # one hot
    prediction_array = predicted_label.detach().cpu().numpy()  # one hot

    if task == "multiclass":
        if len(np.unique(np.argmax(labels_array, 1))) >= 2:
            labels_array = labels_array[:, np.unique(np.argmax(labels_array, 1))]
            prediction_array = prediction_array[
                :, np.unique(np.argmax(labels_array, 1))
            ]
            predicted_probability = predicted_probability[
                :, np.unique(np.argmax(labels_array, 1))
            ]
            predicted_probability = np.array(predicted_probability.detach().cpu())
            auc_list = roc_auc_score(labels_array, predicted_probability, average=None)
            # print('macro auc:', auc_list)
            auc = np.mean(auc_list)

            report = classification_report(
                labels_array, prediction_array, output_dict=True
            )
            recall = report["macro avg"]["recall"]
            precision = report["macro avg"]["precision"]
        else:
            auc = 0
            recall = 0
            precision = 0
            auc_list = []
        correct_label = np.equal(
            np.argmax(labels_array, 1), np.argmax(prediction_array, 1)
        ).sum()
    elif task == "multilabel":
        idx = []
        for l in range(labels_array.shape[1]):  # noqa: E741
            if len(np.unique(labels_array[:, l])) >= 2:
                idx.append(l)
        if len(idx) > 0:
            labels_array = labels_array[:, idx]
            prediction_array = prediction_array[:, idx]
            predicted_probability = predicted_probability[:, idx]
            predicted_probability = np.array(predicted_probability.detach().cpu())
            # print(labels_array, np.any(np.isnan(labels_array)))
            # print(prediction_array, np.any(np.isnan(prediction_array)))
            auc_list = roc_auc_score(labels_array, predicted_probability, average=None)
            auc = np.mean(auc_list)
            report = classification_report(
                labels_array, prediction_array, output_dict=True
            )
            recall = report["macro avg"]["recall"]
            precision = report["macro avg"]["precision"]
        else:
            auc = 0
            recall = 0
            precision = 0
            auc_list = []
        correct_label = np.equal(
            np.argmax(labels_array, 1), np.argmax(prediction_array, 1)
        ).sum()
    return auc, recall, precision, correct_label, auc_list


def test(
    test_loader, model, device, criteria=torch.nn.CrossEntropyLoss(), verbose=True
):
    model.to(device)
    correct_label = 0
    recall_test, precision_test, auc_test = 0, 0, 0
    loss = 0
    total = 0
    auc_test = 0
    model.eval()
    for i, (x, y) in enumerate(test_loader):
        x, y = torch.Tensor(x.float()).to(device), torch.Tensor(y.float()).to(device)
        out = model(x)
        y = y.view(
            y.shape[0],
        )

        label_onehot = torch.zeros(out.shape).to(device)
        pred_onehot = torch.zeros(out.shape).to(device)
        _, predicted_label = out.max(1)
        pred_onehot.scatter_(1, predicted_label.view(-1, 1), 1)

        label_onehot.zero_()
        label_onehot.scatter_(1, y.long().view(-1, 1), 1)

        auc, recall, precision, correct = evaluate(
            label_onehot, pred_onehot, torch.nn.Softmax(-1)(out)
        )

        # prediction = (out > 0.5).view(len(y), ).float()
        # auc, recall, precision, correct = evaluate(y, prediction, out)
        correct_label += correct
        auc_test = auc_test + auc
        recall_test = +recall
        precision_test = +precision
        loss += criteria(out, y.long()).item()
        total += len(x)
    return recall_test, precision_test, auc_test / (i + 1), correct_label, loss


def train(
    train_loader, model, device, optimizer, loss_criterion=torch.nn.CrossEntropyLoss()
):
    model = model.to(device)
    model.train()
    auc_train = 0
    recall_train, precision_train, auc_train, correct_label, epoch_loss = 0, 0, 0, 0, 0
    for i, (signals, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(
            labels.float()
        ).to(device)
        labels = labels.view(
            labels.shape[0],
        )
        labels = labels.view(
            labels.shape[0],
        )
        logits = model(signals)

        label_onehot = torch.zeros(logits.shape).to(device)
        pred_onehot = torch.zeros(logits.shape).to(device)
        _, predicted_label = logits.max(1)
        pred_onehot.scatter_(1, predicted_label.view(-1, 1), 1)

        # labels_th = (labels[:,t]>0.5).float()
        label_onehot.zero_()
        label_onehot.scatter_(1, labels.long().view(-1, 1), 1)

        # auc, recall, precision, correct = evaluate(labels_th.contiguous().view(-1), predicted_label.contiguous().view(-1), predictions.contiguous().view(-1))
        auc, recall, precision, correct = evaluate(
            label_onehot, pred_onehot, torch.nn.Softmax(-1)(logits)
        )

        # auc, recall, precision, correct = evaluate(label_onehot, predicted_label, risks)
        correct_label += correct
        auc_train = auc_train + auc
        recall_train = +recall
        precision_train = +precision

        loss = loss_criterion(logits, labels.long())
        epoch_loss = +loss.item()
        loss.backward()
        optimizer.step()
    return (
        recall_train,
        precision_train,
        auc_train / (i + 1),
        correct_label,
        epoch_loss,
        i + 1,
    )


def train_model(
    model,
    train_loader,
    valid_loader,
    optimizer,
    n_epochs,
    device,
    experiment,
    data="mimic",
    cv=0,
):
    train_loss_trend = []
    test_loss_trend = []

    for epoch in range(n_epochs + 1):
        (
            recall_train,
            precision_train,
            auc_train,
            correct_label_train,
            epoch_loss,
            n_batches,
        ) = train(train_loader, model, device, optimizer)
        recall_test, precision_test, auc_test, correct_label_test, test_loss = test(
            valid_loader, model, device
        )
        train_loss_trend.append(epoch_loss)
        test_loss_trend.append(test_loss)
        if epoch % 10 == 0:
            print("\nEpoch %d" % (epoch))
            print(
                "Training ===>loss: ",
                epoch_loss,
                " Accuracy: %.2f percent"
                % (100 * correct_label_train / (len(train_loader.dataset))),
                " AUC: %.2f" % (auc_train),
            )
            print(
                "Test ===>loss: ",
                test_loss,
                " Accuracy: %.2f percent"
                % (100 * correct_label_test / (len(valid_loader.dataset))),
                " AUC: %.2f" % (auc_test),
            )

    # Save model and results
    if not os.path.exists(os.path.join("./experiments/results/", data)):
        os.mkdir(os.path.join("./experiments/results/", data))

    torch.save(
        model.state_dict(),
        "./experiments/results/" + data + "/" + str(experiment) + "_" + str(cv) + ".pt",
    )
    plt.plot(train_loss_trend, label="Train loss")
    plt.plot(test_loss_trend, label="Validation loss")
    plt.legend()
    # plt.savefig(os.path.join('./plots', data, 'train_loss.pdf'))


def train_model_multiclass(
    model,
    train_loader,
    valid_loader,
    optimizer,
    n_epochs,
    device,
    experiment,
    data="ddg",
    num=5,
    loss_criterion=torch.nn.CrossEntropyLoss(),
    cv=0,
):
    print("Training black-box model on ", data)
    train_loss_trend = []
    test_loss_trend = []

    model.to(device)

    for epoch in range(n_epochs):
        model.train()
        (
            recall_train,
            precision_train,
            auc_train,
            correct_label_train,
            epoch_loss,
            count,
        ) = (0, 0, 0, 0, 0, 0)
        for i, (signals, labels) in enumerate(train_loader):
            signals, labels = signals.to(device), labels.to(device)
            if num > 1:
                # time_points = [int(tt) for tt in np.linspace(20, signals.shape[-1] - 1, num=num)]
                time_points = np.random.randint(
                    low=4, high=signals.shape[-1] - 1, size=num
                )
                time_points = np.sort(time_points)
                time_points[-1] = signals.shape[-1] - 1
            else:
                time_points = [signals.shape[-1] - 1]

            for t in time_points:
                input_signal = signals[:, :, : (t + 1)]

                optimizer.zero_grad()
                predictions = model(input_signal)

                if len(labels.shape) == 3:  # has labels over time
                    label = labels[:, :, t]
                else:
                    label = labels

                label_onehot = label  # assumed already one-hot encoded
                if isinstance(loss_criterion, torch.nn.CrossEntropyLoss):  # noqa: E721
                    pred_onehot = torch.zeros(predictions.shape).to(device)
                    _, predicted_label = predictions.max(1)
                    pred_onehot.scatter_(1, predicted_label.view(-1, 1), 1)
                    task = "multiclass"
                else:
                    pred_onehot = (torch.sigmoid(predictions) > 0.5).float()
                    task = "multilabel"

                auc, recall, precision, correct, auc_list = evaluate_multiclass(
                    label_onehot, pred_onehot, predictions, task=task
                )
                correct_label_train += correct
                auc_train += auc
                recall_train += recall
                precision_train += precision
                count += 1

                if isinstance(loss_criterion, torch.nn.CrossEntropyLoss):  # multiclass
                    # print('here')
                    _, targets = label.max(1)
                    targets = targets.long()
                else:  # multilabel
                    targets = label_onehot
                reconstruction_loss = loss_criterion(predictions, targets)
                epoch_loss += reconstruction_loss.item()
                reconstruction_loss.backward()
                optimizer.step()

        (
            test_loss,
            recall_test,
            precision_test,
            auc_test,
            correct_label_test,
        ) = test_model_multiclass(
            model, valid_loader, num=num, loss_criterion=loss_criterion
        )

        train_loss_trend.append(epoch_loss / ((i + 1) * num))
        test_loss_trend.append(test_loss)

        if epoch % 1 == 0:
            print("\nEpoch %d" % (epoch))
            print(
                "Training ===>loss: ",
                epoch_loss / ((i + 1) * num),
                " Accuracy: %.2f percent"
                % (100 * correct_label_train / (len(train_loader.dataset) * num)),
                " AUC: %.2f" % (auc_train / ((i + 1) * num)),
            )
            print(
                "Test ===>loss: ",
                test_loss,
                " Accuracy: %.2f percent"
                % (100 * correct_label_test / (len(valid_loader.dataset) * num)),
                " AUC: %.2f" % (auc_test),
            )

    (
        test_loss,
        recall_test,
        precision_test,
        auc_test,
        correct_label_test,
    ) = test_model_multiclass(
        model, valid_loader, num=num, loss_criterion=loss_criterion
    )
    print("Test AUC: ", auc_test)

    # Save model and results
    if not os.path.exists(os.path.join("./ckpt/", data)):
        os.mkdir(os.path.join("./ckpt/", data))
    torch.save(
        model.state_dict(),
        "./ckpt/" + data + "/" + str(experiment) + "_" + str(cv) + ".pt",
    )
    plt.plot(train_loss_trend, label="Train loss")
    plt.plot(test_loss_trend, label="Validation loss")
    plt.legend()
    plt.savefig(os.path.join("./plots", data, "train_loss.pdf"))


def train_model_rt(
    model,
    train_loader,
    valid_loader,
    optimizer,
    n_epochs,
    device,
    experiment,
    data="simulation",
    num=5,
    cv=0,
):
    print("Training black-box model on ", data)
    train_loss_trend = []
    test_loss_trend = []

    model.to(device)
    loss_criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        model.train()
        (
            recall_train,
            precision_train,
            auc_train,
            correct_label_train,
            epoch_loss,
            count,
        ) = (0, 0, 0, 0, 0, 0)
        for i, (signals, labels) in enumerate(train_loader):
            # signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(labels.float()).to(device)
            signals, labels = signals.to(device), labels.to(device)
            if "simulation" in data:
                num = 20
                time_points = [
                    int(tt) for tt in np.linspace(1, signals.shape[-1] - 2, num=num)
                ]
            else:
                time_points = [
                    int(tt)
                    for tt in np.logspace(1, np.log10(signals.shape[2] - 1), num=num)
                ]
            for t in time_points:
                input_signal = signals[:, :, : t + 1]
                label = labels[:, t]

                optimizer.zero_grad()
                predictions = model(input_signal)

                label_onehot = torch.zeros(predictions.shape).to(device)
                pred_onehot = torch.zeros(predictions.shape).to(device)
                _, predicted_label = predictions.max(1)
                pred_onehot.scatter_(1, predicted_label.view(-1, 1), 1)

                label_onehot.zero_()
                label_onehot.scatter_(1, label.long().view(-1, 1), 1)

                auc, recall, precision, correct, _ = evaluate_multiclass(
                    label_onehot, pred_onehot, predictions
                )
                correct_label_train += correct
                auc_train += auc
                recall_train += recall
                precision_train += precision
                count += 1

                reconstruction_loss = loss_criterion(predictions, label.long())
                epoch_loss += reconstruction_loss.item()
                reconstruction_loss.backward()
                optimizer.step()

        test_num = num
        (
            test_loss,
            recall_test,
            precision_test,
            auc_test,
            correct_label_test,
        ) = test_model_rt(model, valid_loader, num=test_num)

        train_loss_trend.append(epoch_loss / ((i + 1) * num))
        test_loss_trend.append(test_loss)

        if epoch % 10 == 0:
            print("\nEpoch %d" % (epoch))
            print(
                "Training ===>loss: ",
                epoch_loss / ((i + 1) * num),
                " Accuracy: %.2f percent"
                % (100 * correct_label_train / (len(train_loader.dataset) * num)),
                " AUC: %.2f" % (auc_train / ((i + 1) * num)),
            )
            print(
                "Test ===>loss: ",
                test_loss,
                " Accuracy: %.2f percent"
                % (100 * correct_label_test / (len(valid_loader.dataset) * test_num)),
                " AUC: %.2f" % (auc_test),
            )

    (
        test_loss,
        recall_test,
        precision_test,
        auc_test,
        correct_label_test,
    ) = test_model_rt(model, valid_loader, num=test_num)
    print("Test AUC: ", auc_test)

    # Save model and results
    if not os.path.exists(os.path.join("./experiments/results/", data)):
        os.mkdir(os.path.join("./experiments/results/", data))
    torch.save(
        model.state_dict(),
        "./experiments/results/" + data + "/" + str(experiment) + "_" + str(cv) + ".pt",
    )
    plt.plot(train_loss_trend, label="Train loss")
    plt.plot(test_loss_trend, label="Validation loss")
    plt.legend()
    # plt.savefig(os.path.join('./plots', data, 'train_loss.pdf'))


def train_model_rt_binary(
    model,
    train_loader,
    valid_loader,
    optimizer,
    n_epochs,
    device,
    experiment,
    data="simulation",
    num=5,
    cv=0,
):
    train_loss_trend = []
    test_loss_trend = []

    model.to(device)
    loss_criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(n_epochs):
        model.train()
        (
            recall_train,
            precision_train,
            auc_train,
            correct_label_train,
            epoch_loss,
            count,
        ) = (0, 0, 0, 0, 0, 0)
        for i, (signals, labels) in enumerate(train_loader):
            signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(
                labels.float()
            ).to(device)
            if data == "simulation":
                time_points = [
                    int(tt) for tt in np.linspace(1, signals.shape[2] - 2, num=num)
                ]
            else:
                time_points = [
                    int(tt)
                    for tt in np.logspace(0, np.log10(signals.shape[2] - 1), num=num)
                ]

            for t in time_points:
                predictions_logits = model(signals[:, :, : t + 1])
                predictions = torch.sigmoid(predictions_logits)
                predicted_label = (predictions > 0.5).float()
                labels_th = labels[:, t].float()
                auc, recall, precision, correct = evaluate_binary(
                    labels_th.contiguous().view(-1),
                    predicted_label.contiguous().view(-1),
                    predictions.contiguous().view(-1),
                )
                correct_label_train += correct
                auc_train += auc
                recall_train += recall
                precision_train += precision
                count += 1
                optimizer.zero_grad()
                reconstruction_loss = loss_criterion(
                    predictions_logits, labels[:, t].view(-1, 1).to(device)
                )
                epoch_loss += reconstruction_loss.item()
                reconstruction_loss.backward()
                optimizer.step()

        test_num = num
        (
            test_loss,
            recall_test,
            precision_test,
            auc_test,
            correct_label_test,
        ) = test_model_rt_binary(model, valid_loader, num=test_num)

        train_loss_trend.append(epoch_loss / ((i + 1) * num))
        test_loss_trend.append(test_loss)

        if epoch % 10 == 0:
            print("\nEpoch %d" % (epoch))
            print(
                "Training ===>loss: ",
                epoch_loss / ((i + 1) * num),
                " Accuracy: %.2f percent"
                % (100 * correct_label_train / (len(train_loader.dataset) * num)),
                " AUC: %.2f" % (auc_train / ((i + 1) * num)),
            )
            print(
                "Test ===>loss: ",
                test_loss,
                " Accuracy: %.2f percent"
                % (100 * correct_label_test / (len(valid_loader.dataset) * test_num)),
                " AUC: %.2f" % (auc_test),
            )

    (
        test_loss,
        recall_test,
        precision_test,
        auc_test,
        correct_label_test,
    ) = test_model_rt_binary(model, valid_loader)
    print("Test loss: ", test_loss)

    # Save model and results
    if not os.path.exists(os.path.join("./ckpt/", data)):
        os.mkdir(os.path.join("./ckpt/", data))
    torch.save(
        model.state_dict(),
        "./ckpt/" + data + "/" + str(experiment) + "_" + str(cv) + ".pt",
    )
    plt.plot(train_loss_trend, label="Train loss")
    plt.plot(test_loss_trend, label="Validation loss")
    plt.legend()
    plt.savefig(os.path.join("./plots", data, "train_loss.pdf"))


def train_model_rt_rg(
    model,
    train_loader,
    valid_loader,
    optimizer,
    n_epochs,
    device,
    experiment,
    data="ghg",
):
    print("training data: ", data)
    train_loss_trend = []
    test_loss_trend = []

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    loss_criterion = torch.nn.MSELoss()
    print("loss function: MSE")
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        num = 20
        for i, (signals, labels) in enumerate(train_loader):
            signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(
                labels.float()
            ).to(device)
            for t in [int(tt) for tt in np.linspace(0, signals.shape[2] - 1, num=num)]:
                optimizer.zero_grad()
                predictions = model(signals[:, :, : t + 1])

                reconstruction_loss = loss_criterion(
                    predictions, labels[:, t].to(device)
                )
                epoch_loss += reconstruction_loss.item()
                reconstruction_loss.backward()
                optimizer.step()

        test_loss = test_model_rt_rg(model, valid_loader)
        train_loss_trend.append(epoch_loss / (num * (i + 1)))
        test_loss_trend.append(test_loss)

        if epoch % 10 == 0:
            print("\nEpoch %d" % (epoch))
            print("Training ===>loss: ", epoch_loss / (num * (i + 1)))
            print("Test ===>loss: ", test_loss)

    test_loss = test_model_rt_rg(model, valid_loader)
    print("Test loss: ", test_loss)

    # Save model and results
    if not os.path.exists(os.path.join("./experiments/", data)):
        os.mkdir(os.path.join("./ckpt/", data))
    torch.save(model.state_dict(), "./ckpt/" + data + "/" + str(experiment) + ".pt")
    plt.plot(train_loss_trend, label="Train loss")
    plt.plot(test_loss_trend, label="Validation loss")
    plt.legend()
    plt.savefig(os.path.join("./plots", data, "train_loss.png"))


def test_model_rt(model, test_loader, num=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    correct_label_test = 0
    recall_test, precision_test, auc_test = 0, 0, 0
    count = 0
    test_loss = 0
    for i, (signals, labels) in enumerate(test_loader):
        signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(
            labels.float()
        ).to(device)
        for t in [int(tt) for tt in np.linspace(0, signals.shape[2] - 2, num=num)]:
            label = labels[:, t].view(-1, 1)
            label_onehot = torch.FloatTensor(label.shape[0], 2).to(device)
            pred_onehot = torch.FloatTensor(label.shape[0], 2).to(device)
            preds = model(signals[:, :, : t + 1])

            if preds.shape[1] == 1 and 0:
                predictions = torch.cuda.FloatTensor(preds.shape[0], 2).fill_(0)
                predictions[:, 1] = preds[:, 0]
                predictions[:, 0] = 1 - preds[:, 0]
            else:
                predictions = preds

            _, predicted_label = predictions.max(1)
            pred_onehot.zero_()
            pred_onehot.scatter_(1, predicted_label.view(-1, 1), 1)
            label_onehot.zero_()
            label_onehot.scatter_(1, labels[:, t].long().view(-1, 1), 1)
            auc, recall, precision, correct, _ = evaluate_multiclass(
                label_onehot, pred_onehot, predictions
            )
            correct_label_test += correct
            auc_test += auc
            recall_test += recall
            precision_test += precision
            count += 1
            loss_criterion = torch.nn.CrossEntropyLoss()
            loss = loss_criterion(predictions, labels[:, t].long().to(device))
            test_loss += loss.item()

    test_loss = test_loss / ((i + 1) * num)
    return (
        test_loss,
        recall_test,
        precision_test,
        auc_test / ((i + 1) * num),
        correct_label_test,
    )


def test_model_rt_binary(model, test_loader, num=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    correct_label_test = 0
    recall_test, precision_test, auc_test = 0, 0, 0
    count = 0
    test_loss = 0
    for i, (signals, labels) in enumerate(test_loader):
        signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(
            labels.float()
        ).to(device)
        for t in [int(tt) for tt in np.linspace(0, signals.shape[2] - 2, num=num)]:
            # for t in [24]:
            prediction_logits = model(signals[:, :, : t + 1])
            prediction = torch.sigmoid(prediction_logits)
            predicted_label = (prediction > 0.5).float()
            labels_th = labels[:, t].float()
            auc, recall, precision, correct = evaluate_binary(
                labels_th.contiguous().view(-1),
                predicted_label.contiguous().view(-1),
                prediction.contiguous().view(-1),
            )
            correct_label_test += correct
            auc_test += auc
            recall_test += recall
            precision_test += precision
            count += 1
            loss = torch.nn.BCEWithLogitsLoss()(
                prediction_logits, labels[:, t].view(-1, 1).to(device)
            )
            test_loss += loss.item()

    test_loss = test_loss / ((i + 1) * num)
    return (
        test_loss,
        recall_test,
        precision_test,
        auc_test / ((i + 1) * num),
        correct_label_test,
    )


def test_model_rt_rg(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    test_loss = 0
    num = 50
    for i, (signals, labels) in enumerate(test_loader):
        signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(
            labels.float()
        ).to(device)
        for t in [int(tt) for tt in np.linspace(0, signals.shape[2] - 1, num=num)]:
            prediction = model(signals[:, :, : t + 1])
            loss = torch.nn.MSELoss()(prediction, labels[:, t].to(device))
            test_loss += loss.item()

    test_loss = test_loss / (num * (i + 1))
    return test_loss


def test_model_multiclass(
    model, test_loader, num=5, loss_criterion=torch.nn.CrossEntropyLoss()
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    correct_label_test = 0
    recall_test, precision_test, auc_test = 0, 0, 0
    count = 0
    test_loss = 0
    auc_class_list = []
    for i, (signals, labels) in enumerate(test_loader):
        signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(
            labels.float()
        ).to(device)
        if num > 1:
            time_points = [
                int(tt) for tt in np.linspace(20, signals.shape[-1] - 2, num=num)
            ]
        else:
            time_points = [signals.shape[2] - 1]
        for t in time_points:
            if len(labels.shape) == 3:  # has labels over time
                label = labels[:, :, t]
            else:
                label = labels

            label_onehot = label
            predictions = model(signals)

            # print(loss_criterion)
            if isinstance(loss_criterion, torch.nn.CrossEntropyLoss):  # noqa: E721
                pred_onehot = torch.FloatTensor(labels.shape[0], labels.shape[1]).to(
                    device
                )
                _, predicted_label = predictions.max(1)
                pred_onehot.zero_()
                pred_onehot.scatter_(1, predicted_label.view(-1, 1), 1)
                task = "multiclass"
            else:
                pred_onehot = (torch.sigmoid(predictions) > 0.5).float()
                task = "multilabel"

            auc, recall, precision, correct, auc_list = evaluate_multiclass(
                label_onehot, pred_onehot, torch.sigmoid(predictions), task=task
            )
            auc_class_list.append(auc_list)
            correct_label_test += correct
            auc_test += auc
            recall_test += recall
            precision_test += precision
            count += 1
            if isinstance(loss_criterion, torch.nn.CrossEntropyLoss):  # noqa: E721
                _, targets = label.max(1)
                targets = targets.long()
            else:
                targets = label
            loss = loss_criterion(predictions, targets)
            test_loss += loss.item()

    test_loss = test_loss / ((i + 1) * num)
    auc_class_list = np.array(auc_class_list).sum(0)
    print("class auc:", auc_class_list / ((i + 1) * num))
    return (
        test_loss,
        recall_test,
        precision_test,
        auc_test / ((i + 1) * num),
        correct_label_test,
    )


def train_reconstruction(
    model, train_loader, valid_loader, n_epochs, device, experiment
):
    train_loss_trend = []
    test_loss_trend = []
    model.to(device)

    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=0.0001, weight_decay=1e-3)
    # optimizer = torch.optim.SGD(parameters, lr=0.001, momentum=0.9, weight_decay=1e-4)

    for epoch in range(n_epochs + 1):
        model.train()
        epoch_loss = 0
        for i, (signals, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            signals, _ = torch.Tensor(signals.float()).to(device), torch.Tensor(
                labels.float()
            ).to(device)
            mu, logvar, z = model.encode(signals)
            recon = model.decode(z)
            loss = torch.nn.MSELoss()(
                recon, signals[:, :, -1].view(len(signals), -1)
            ) - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            epoch_loss = +loss.item()
            loss.backward()
            optimizer.step()
        test_loss = test_reconstruction(model, valid_loader, device)

        train_loss_trend.append(epoch_loss)
        test_loss_trend.append(test_loss)
        if epoch % 10 == 0:
            print("\nEpoch %d" % (epoch))
            print("Training ===>loss: ", epoch_loss)
            print("Test ===>loss: ", test_loss)

    # Save model and results
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")
    torch.save(model.state_dict(), "./ckpt/" + str(experiment) + ".pt")
    plt.plot(train_loss_trend, label="Train loss")
    plt.plot(test_loss_trend, label="Validation loss")
    plt.legend()
    plt.savefig("train_loss.pdf")


def test_reconstruction(model, valid_loader, device):
    model.eval()
    test_loss = 0
    for i, (signals, labels) in enumerate(valid_loader):
        signals, _ = torch.Tensor(signals.float()).to(device), torch.Tensor(
            labels.float()
        ).to(device)
        mu, logvar, z = model.encode(signals)
        recon = model.decode(z)
        loss = torch.nn.MSELoss()(
            recon, signals[:, :, -1].view(len(signals), -1)
        ) - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        test_loss = +loss.item()
    return test_loss


def load_data(batch_size, path="./data/", **kwargs):
    transform = kwargs["transform"] if "transform" in kwargs.keys() else "normalize"
    task = kwargs["task"] if "task" in kwargs.keys() else "mortality"
    p_data = PatientData(path, task=task, shuffle=False, transform=transform)
    test_bs = kwargs["test_bs"] if "test_bs" in kwargs.keys() else None

    features = (
        kwargs["features"]
        if "features" in kwargs.keys()
        else range(p_data.train_data.shape[1])
    )
    p_data.train_data = p_data.train_data[:, features, :]
    p_data.test_data = p_data.test_data[:, features, :]

    p_data.feature_size = len(features)
    n_train = int(0.9 * p_data.train_data.shape[0])
    if "cv" in kwargs.keys():
        if task == "mortality":
            kf = KFold(n_splits=5, random_state=42)
            train_idx, valid_idx = list(kf.split(p_data.train_data))[kwargs["cv"]]
        else:
            sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=88)
            train_idx, valid_idx = list(
                sss.split(p_data.train_data[:, :, -1], p_data.train_label[:, :, -1])
            )[kwargs["cv"]]
    else:
        if task == "mortality":
            train_idx = range(n_train)
            valid_idx = range(n_train, p_data.n_train)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=88)
            train_idx, valid_idx = list(
                sss.split(p_data.train_data[:, :, -1], p_data.train_label[:, :, -1])
            )[0]

    train_dataset = utils.TensorDataset(
        torch.Tensor(p_data.train_data[train_idx, :, :]),
        torch.Tensor(p_data.train_label[train_idx]),
    )

    valid_dataset = utils.TensorDataset(
        torch.Tensor(p_data.train_data[valid_idx, :, :]),
        torch.Tensor(p_data.train_label[valid_idx]),
    )
    test_dataset = utils.TensorDataset(
        torch.Tensor(p_data.test_data), torch.Tensor(p_data.test_label)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size
    )  # p_data.n_train - int(0.8 * p_data.n_train))

    if test_bs is not None:
        test_loader = DataLoader(test_dataset, batch_size=test_bs)
    else:
        test_loader = DataLoader(test_dataset, batch_size=len(p_data.test_data))

    if task == "mortality":
        print(
            "Train set: ",
            np.count_nonzero(p_data.train_label[0 : int(0.8 * p_data.n_train)]),
            "patient who died out of %d total" % (int(0.8 * p_data.n_train)),
            "(Average missing in train: %.2f)"
            % (np.mean(p_data.train_missing[0 : int(0.8 * p_data.n_train)])),
        )
        print(
            "Valid set: ",
            np.count_nonzero(p_data.train_label[int(0.8 * p_data.n_train) :]),
            "patient who died out of %d total"
            % (len(p_data.train_label[int(0.8 * p_data.n_train) :])),
            "(Average missing in validation: %.2f)"
            % (np.mean(p_data.train_missing[int(0.8 * p_data.n_train) :])),
        )
        print(
            "Test set: ",
            np.count_nonzero(p_data.test_label),
            "patient who died  out of %d total" % (len(p_data.test_data)),
            "(Average missing in test: %.2f)" % (np.mean(p_data.test_missing)),
        )
    return p_data, train_loader, valid_loader, test_loader


# def load_ghg_data(batch_size, path="./data_generator/data", **kwargs):
#     p_data = GHGData(path, transform=None)  # data already normalized zero mean 1 std
#     # print('ghg label stats', np.mean(p_data.train_label),np.std(p_data.train_label))
#     features = kwargs["features"] if "features" in kwargs.keys() else range(p_data.train_data.shape[1])
#     p_data.train_data = p_data.train_data[:, features, :]
#     p_data.test_data = p_data.test_data[:, features, :]

#     n_train = int(0.8 * len(x_train))
#     if "cv" in kwargs.keys():
#         kf = KFold(n_splits=5, random_state=42)
#         train_idx, valid_idx = list(kf.split(x_train))[kwargs["cv"]]
#     else:
#         train_idx = range(n_train)
#         valid_idx = range(ntrain.len(x_train))

#     train_dataset = utils.TensorDataset(
#         torch.Tensor(p_data.train_data[train_idx, :, :]), torch.Tensor(p_data.train_label[train_idx])
#     )
#     valid_dataset = utils.TensorDataset(
#         torch.Tensor(p_data.train_data[train_idx, :, :]), torch.Tensor(p_data.train_label[train_idx])
#     )
#     test_dataset = utils.TensorDataset(torch.Tensor(p_data.test_data[:, :, :]), torch.Tensor(p_data.test_label))
#     train_loader = DataLoader(train_dataset, batch_size=batch_size)
#     valid_loader = DataLoader(valid_dataset, batch_size=p_data.n_train - int(0.8 * p_data.n_train))
#     test_loader = DataLoader(test_dataset, batch_size=p_data.n_test)
#     # print('Train set: ', p_data.train_data.shape)
#     # print('Valid set: ', p_data.train_data.shape)
#     # print('Test set: ', p_data.test_data.shape)
#     p_data.feature_size = len(features)
#     return p_data, train_loader, valid_loader, test_loader


def load_simulated_data(
    batch_size=100, datapath="./data/state", data_type="state", percentage=1.0, **kwargs
):
    if data_type == "state":
        file_name = "state_dataset_"
    else:
        file_name = ""
    with open(os.path.join(datapath, file_name + "x_train.pkl"), "rb") as f:
        x_train = pkl.load(f)
    with open(os.path.join(datapath, file_name + "y_train.pkl"), "rb") as f:
        y_train = pkl.load(f)
    with open(os.path.join(datapath, file_name + "x_test.pkl"), "rb") as f:
        x_test = pkl.load(f)
    with open(os.path.join(datapath, file_name + "y_test.pkl"), "rb") as f:
        y_test = pkl.load(f)

    features = (
        kwargs["features"]
        if "features" in kwargs.keys()
        else list(range(x_test.shape[1]))
    )
    test_bs = kwargs["test_bs"] if "test_bs" in kwargs.keys() else None

    perm = np.random.permutation(x_train.shape[0])
    x_train = x_train[perm]
    y_train = y_train[perm]

    total_sample_n = int(len(x_train) * percentage)
    x_train = x_train[:total_sample_n]
    y_train = y_train[:total_sample_n]
    n_train = int(0.8 * len(x_train))
    x_train = x_train[:, features, :]
    x_test = x_test[:, features, :]

    n_train = int(0.8 * len(x_train))
    if "cv" in kwargs.keys():
        print("cv : ", kwargs["cv"])
        kf = KFold(n_splits=5, random_state=88)
        train_idx, valid_idx = list(kf.split(x_train))[kwargs["cv"]]
    else:
        train_idx = range(n_train)
        valid_idx = range(n_train, len(x_train))

    train_dataset = utils.TensorDataset(
        torch.Tensor(x_train[train_idx, :, :]), torch.Tensor(y_train[train_idx, :])
    )
    valid_dataset = utils.TensorDataset(
        torch.Tensor(x_train[valid_idx, :, :]), torch.Tensor(y_train[valid_idx, :])
    )

    test_dataset = utils.TensorDataset(
        torch.Tensor(x_test[:, :, :]), torch.Tensor(y_test)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(
        valid_dataset, batch_size=len(x_train) - int(0.8 * n_train)
    )
    if test_bs is not None:
        test_loader = DataLoader(test_dataset, batch_size=test_bs)
    else:
        test_loader = DataLoader(test_dataset, batch_size=len(x_test))
    return np.concatenate([x_train, x_test]), train_loader, valid_loader, test_loader


def logistic(x):
    return 1.0 / (1 + np.exp(-1 * x))


def top_risk_change(exp):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    span = []
    testset = list(exp.test_loader.dataset)
    for i, (signal, label) in enumerate(testset):
        exp.risk_predictor.load_state_dict(torch.load("./ckpt/mimic/risk_predictor.pt"))
        exp.risk_predictor.to(device)
        exp.risk_predictor.eval()
        risk = []
        for t in range(1, 48):
            risk.append(
                exp.risk_predictor(
                    signal[:, 0:t].view(1, signal.shape[0], t).to(device)
                ).item()
            )
        span.append((i, max(risk) - min(risk)))
    span.sort(key=lambda pair: pair[1], reverse=True)
    print([x[0] for x in span[0:300]])


def test_cond(mean, covariance, sig_ind, x_ind):
    x_ind = x_ind.unsqueeze(-1)
    mean_1 = torch.cat((mean[:, :sig_ind], mean[:, sig_ind + 1 :]), 1).unsqueeze(-1)
    cov_1_2 = torch.cat(
        ([covariance[:, 0:sig_ind, sig_ind], covariance[:, sig_ind + 1 :, sig_ind]]), 1
    ).unsqueeze(-1)
    cov_2_2 = covariance[:, sig_ind, sig_ind]
    cov_1_1 = torch.cat(
        ([covariance[:, 0:sig_ind, :], covariance[:, sig_ind + 1 :, :]]), 1
    )
    cov_1_1 = torch.cat(([cov_1_1[:, :, 0:sig_ind], cov_1_1[:, :, sig_ind + 1 :]]), 2)
    mean_cond = (
        mean_1 + torch.bmm(cov_1_2, (x_ind - mean[:, sig_ind]).unsqueeze(-1)) / cov_2_2
    )
    covariance_cond = (
        cov_1_1 - torch.bmm(cov_1_2, torch.transpose(cov_1_2, 2, 1)) / cov_2_2
    )
    return mean_cond, covariance_cond


def shade_state_state_data(state_subj, t, ax, data="simulation"):
    # Shade the state on simulation data plots
    for ttt in range(t[0], len(t)):
        if state_subj[ttt] == 0:
            ax.axvspan(ttt + 1, ttt, facecolor="blue", alpha=0.3)
        elif state_subj[ttt] == 1:
            ax.axvspan(ttt + 1, ttt, facecolor="green", alpha=0.3)
        elif state_subj[ttt] == 2:
            ax.axvspan(ttt + 1, ttt, facecolor="orange", alpha=0.3)


def shade_state(gt_importance_subj, t, ax, data="simulation"):
    # Shade the state on simulation data plots
    if gt_importance_subj.shape[0] >= 3:
        gt_importance_subj = gt_importance_subj.transpose(1, 0)

    if not data == "simulation_spike":
        prev_color = (
            "g"
            if np.argmax(gt_importance_subj[:, 1]) < np.argmax(gt_importance_subj[:, 2])
            else "y"
        )
        print("######################", t[1])
        for ttt in range(t[1], t[-1]):
            # state = np.argmax(gt_importance_subj[ttt, :])
            # ax.axvspan(ttt - 1, ttt, facecolor=cmap(state), alpha=0.3)
            if gt_importance_subj[ttt, 1] == 1:
                ax.axvspan(ttt - 1, ttt, facecolor="g", alpha=0.3)
                prev_color = "g"
            elif gt_importance_subj[ttt, 2] == 1:
                ax.axvspan(ttt - 1, ttt, facecolor="y", alpha=0.3)
                prev_color = "y"
            elif not prev_color is None:  # noqa: E714
                ax.axvspan(ttt - 1, ttt, facecolor=prev_color, alpha=0.3)


def plot_importance(
    subject,
    signals,
    label,
    a,
    a_std,
    a_max,
    n_feats_to_plot,
    signals_to_analyze,
    color_map,
    fmap,
    data,
    gt_importance_subj,
    save_path,
    patient_data,
):
    important_signals = []
    markers = ["*", "D", "X", "o", "8", "v", "+"]
    f, axs = plt.subplots(6)
    t = np.arange(1, a[0].shape[-1] + 1)

    # a[0] = 2./(1.+np.exp(-3*a[0])) - 1.
    if hasattr(patient_data, "test_intervention"):
        f_color = ["g", "b", "r", "c", "m", "y", "k"]
        for int_ind, intervention in enumerate(
            patient_data.test_intervention[subject, :, :]
        ):
            if sum(intervention) != 0:
                switch_point = []
                intervention = intervention[1:]
                for i in range(1, len(intervention)):
                    if intervention[i] != intervention[i - 1]:
                        switch_point.append(i)
                if len(switch_point) % 2 == 1:
                    switch_point.append(len(intervention) - 1)
                for count in range(int(len(switch_point) / 2)):
                    if count == 0:
                        axs[0].axvspan(
                            xmin=switch_point[count * 2],
                            xmax=switch_point[2 * count + 1],
                            facecolor=f_color[int_ind % len(f_color)],
                            alpha=0.2,
                            label="%s" % (intervention_list[int_ind]),
                        )
                    else:
                        axs[0].axvspan(
                            xmin=switch_point[count * 2],
                            xmax=switch_point[2 * count + 1],
                            facecolor=f_color[int_ind % len(f_color)],
                            alpha=0.2,
                        )

    if not gt_importance_subj is None:  # noqa: E714
        shade_state(gt_importance_subj, t, axs[0], data)

    for ax_ind, ax in enumerate(axs[1:]):
        ax.grid()
        ax.tick_params(axis="both", labelsize=36)
        ax.set_ylabel("importance", fontweight="bold", fontsize=32)
        # if ax_ind in [0, 1, 2]:
        #     ax.set_ylim(bottom=-0.02, top=1.)

        for ind, sig in a_max[ax_ind][0:n_feats_to_plot]:
            ind = int(ind)
            ref_ind = signals_to_analyze[ind]
            if ref_ind not in important_signals:
                important_signals.append(ref_ind)
            c = color_map[ref_ind]
            if np.sum(a_std[ax_ind][ind]) > 0:
                ax.errorbar(
                    t,
                    a[ax_ind][ind, :],
                    yerr=a_std[ax_ind][ind, :],
                    marker=markers[
                        list(important_signals).index(ref_ind) % len(markers)
                    ],
                    linewidth=3,
                    elinewidth=1,
                    markersize=9,
                    markeredgecolor="k",
                    color=c,
                    label="%s" % (fmap[ref_ind]),
                )
            else:
                ax.errorbar(
                    t,
                    a[ax_ind][ind, :],
                    yerr=a_std[ax_ind][ind, :],
                    linewidth=3,
                    elinewidth=1,
                    color=c,
                    label="%s" % (fmap[ref_ind]),
                )
    important_signals = np.unique(important_signals)
    max_plot = (torch.max(torch.abs(signals[important_signals, :]))).item()
    for i, ref_ind in enumerate(important_signals):
        c = color_map[ref_ind]
        if data == "mimic":
            axs[0].plot(
                np.array(signals[ref_ind, 1:]) / max_plot,
                linewidth=3,
                color=c,
                label="%s" % (fmap[ref_ind]),
            )
        else:
            axs[0].plot(
                np.array(signals[ref_ind, 1:]),
                linewidth=3,
                color=c,
                label="%s" % (fmap[ref_ind]),
            )
    axs[0].tick_params(axis="both", labelsize=36)
    axs[0].grid()

    axs[0].plot(np.array(label), "--", linewidth=6, label="Risk score", c="black")
    axs[0].set_title(
        "Time series signals and Model's predicted risk", fontweight="bold", fontsize=40
    )
    axs[1].set_title("Feature importance FIT", fontweight="bold", fontsize=40)
    axs[2].set_title("Feature importance AFO", fontweight="bold", fontsize=40)
    axs[3].set_title("Feature importance FO", fontweight="bold", fontsize=40)
    axs[4].set_title(
        "Feature importance Sensitivity analysis", fontweight="bold", fontsize=40
    )
    axs[5].set_title("Feature importance Attention", fontweight="bold", fontsize=40)
    axs[5].set_xlabel("time", fontweight="bold", fontsize=32)
    axs[0].set_ylabel("signal value", fontweight="bold", fontsize=32)

    f.set_figheight(40)
    f.set_figwidth(60)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(
        os.path.join(save_path, "feature_%d.pdf" % (subject)),
        dpi=300,
        orientation="landscape",
    )
    fig_legend = plt.figure(figsize=(13, 1.2))
    handles, labels = axs[0].get_legend_handles_labels()
    plt.figlegend(
        handles,
        labels,
        loc="upper left",
        ncol=4,
        fancybox=True,
        handlelength=6,
        fontsize="xx-large",
    )
    fig_legend.savefig(
        os.path.join(save_path, "legend_%d.pdf" % subject), dpi=300, bbox_inches="tight"
    )

    for imp_plot_ind in range(4):
        heatmap_fig = plt.figure(figsize=(15, 1) if data == "simulation" else (16, 9))
        plt.yticks(rotation=0)
        sns.heatmap(
            a[imp_plot_ind], yticklabels=fmap, square=True if data == "mimic" else False
        )  # , vmin=0, vmax=1)
        heatmap_fig.savefig(
            os.path.join(
                save_path,
                "heatmap_%s_%s.pdf"
                % (str(subject), ["FIT", "AFO", "FO", "Sens"][imp_plot_ind]),
            )
        )
    if data == "simulation":
        heatmap_gt = plt.figure(figsize=(20, 1))
        plt.yticks(rotation=0)
        sns.heatmap(gt_importance_subj, yticklabels=fmap)
        heatmap_gt.savefig(
            os.path.join(save_path, "heatmap_%s_ground_truth.pdf" % str(subject))
        )

    return important_signals


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_median_rank(ranked_feats, ground_truth, soft=False, K=4, tau=0.2):
    # n x d x t - size of both tensors
    median_ranks = np.empty((ranked_feats.shape[0], ranked_feats.shape[2]))
    median_ranks[:] = np.NaN
    if soft:
        for n in range(ranked_feats.shape[0]):
            curr_sample = ranked_feats[n]
            for t in range(ranked_feats.shape[2]):
                idx = np.where(ground_truth[n, :, t] > tau)[0]
                if len(idx) > 0:
                    median_ranks[n, t] = np.median(curr_sample[:, t])
    else:
        for n in range(ranked_feats.shape[0]):
            curr_sample = ranked_feats[n]
            for t in range(ranked_feats.shape[2]):
                idx = np.where(ground_truth[n, :, t])[0]
                if len(idx) > 0:
                    median_ranks[n, t] = np.median(curr_sample[:, t])
    return median_ranks, np.nanmean(median_ranks), np.nanstd(median_ranks)


def plot_heatmap_text(ranked_scores, scores, filepath, ax):
    # assumes same shape of ranked scores and scores
    ax.pcolormesh(-ranked_scores, cmap=matplotlib.cm.Greens)
    for y in range(ranked_scores.shape[0]):
        for x in range(ranked_scores.shape[1]):
            ax.text(
                x + 0.5,
                y + 0.5,
                "%d" % ranked_scores[y, x],
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=20,
                color="r",
                fontweight="bold",
            )

    # ax.colorbar(heatmap)
    # plt.savefig(filepath, dpi=300, bbox_inches='tight')
