from Code.engine.preprocessing import rolling_forecast_origin_split, dataloader,feature_1_denormalize,feature_1_normalize
from Code.engine.utilities import move_on, batch_window_generator
from torchmetrics.regression import MeanAbsolutePercentageError
from Code.engine.targets_plot_generator import run_TPG_link
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime
import torch.optim as optim
from torch import nn
import numpy as np
import optuna
import random
import torch
import os


#Fixed Seed :
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)



def train_model_loop(model: nn.Module, training_data, scaler_1, scaler_2, epochs: int=10, batch_size: int = 32, lr: float = 0.0001, l2_lambda: float = 0.01, window_size: int = 8, patience: int = 5, alpha = 0.5):
    #We are Troubleshooting Train Model for Inconsistency with Different MLP models.
    #Recording variables
    per_epoch_loss = []
    all_time_loss =[]
    prediction = None
    patience_counter = 0

    #Loss Function criteria
    criterion = nn.MSELoss()
    mape = MeanAbsolutePercentageError()


    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    best_val_loss = np.finfo(np.float32).max

    for epoch in range(epochs):


        if patience_counter==patience:
            return model, all_time_loss, per_epoch_loss, prediction
        prediction = None
        # print(f"Training model in epoch {epoch+1} out of {epochs}")
        epoch_loss = 0
        batch_window_tuple_list = batch_window_generator(training_data, batch_size=batch_size, window_size=window_size)

        for data_tuple in batch_window_tuple_list:
            batch, window = data_tuple
            optimizer.zero_grad()
            batch_prediction = []

            if len(batch) == 0:
                continue

            for i in range(len(batch)):
                input_window = move_on(batch, window, i)
                input_window = input_window.unsqueeze(0)
                pred = model(input_window)
                batch_prediction.append(pred)


            batch_prediction=[pred.item() for pred in batch_prediction]

            batch_prediction = feature_1_denormalize(batch_prediction, scaler_1= scaler_1, scaler_2 = scaler_2)
            batch = feature_1_denormalize(batch, scaler_1= scaler_1, scaler_2 =scaler_2)

            batch_prediction = [torch.tensor(pred, dtype=torch.float32) for pred in batch_prediction]

            # #Checking the data type of batch_prediction & batch
            # print(f"The data type of batch_prediction is list of {batch_prediction[0].dtype}")
            # print(f"The data type of batch io a list of {batch[0].dtype}")
            # print("----------End--------------")

            # 1. Convert list of torch.float32 tensors to one tensor
            batch_prediction_tensor = torch.stack(batch_prediction)  # shape: [batch_size], dtype: float32

            # 2. Convert list of float64 Python floats to tensor, then cast to float32
            batch_targets_tensor = torch.tensor(batch, dtype=torch.float64).to(torch.float32)  # shape: [batch_size]

            # Optional: reshape if needed to match prediction shape, e.g. [batch_size, 1]
            batch_targets_tensor = batch_targets_tensor.reshape(-1, 1)
            batch_prediction_tensor = batch_prediction_tensor.reshape(-1, 1)



            #loss calculation ; l2 + base
            batch_base_loss_mse = criterion(batch_prediction_tensor, batch_targets_tensor) * alpha
            batch_base_loss_mape = mape(batch_prediction_tensor, batch_targets_tensor)* (1-alpha)
            l2_loss = sum(param.pow(2).sum() for param in model.parameters())
            loss = batch_base_loss_mse + batch_base_loss_mape  + l2_loss* l2_lambda

            #Update part
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            #Recording
            epoch_loss+=loss.item()
            all_time_loss.append(loss.item())

            batch_np = batch_prediction_tensor.detach().cpu().numpy()
            prediction = batch_np if prediction is None else np.vstack((prediction, batch_np))


        if epoch_loss< best_val_loss:
            best_val_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter+=1
        per_epoch_loss.append(epoch_loss)
        scheduler.step(epoch_loss)
        # print(f"Done Training epoch {epoch} out of {epochs} with error {epoch_loss}")

    return  model, all_time_loss, per_epoch_loss, prediction

def train_and_save_results(MLP_model_class,hidden_size, input_size, epochs: int=40, batch_size: int = 32, lr: float = 0.0001, l2_lambda: float = 0.01, patience: int = 10, alpha = 0.5, folder_name= None):

    model = MLP_model_class(hidden_size = hidden_size,input_feature= input_size)
    train_data, test_data, scaler_1, scaler_2 = dataloader()
    data = np.concatenate((train_data, test_data), axis=0)

    model, _, _, _ = train_model_loop(model=model, training_data= data, epochs=epochs, batch_size=batch_size,
                                    lr= lr,l2_lambda= l2_lambda,window_size= input_size,patience=patience,
                                    alpha= alpha, scaler_1=scaler_1, scaler_2=scaler_2)
    testing_loss, _  = evaluate(model, test_data, batch_size, input_size, scaler_1, scaler_2)
    _, prediction    = evaluate(model, data, batch_size, input_size ,scaler_1, scaler_2)

    # print(f"The prediction output of the Model is the following: ")
    # np.savetxt('prediction.csv', prediction, delimiter=',')

    predicted_target_numpy = []
    for phase_list in prediction:
        scalar_values = [t.item() for t in phase_list]
        predicted_target_numpy.append(np.array(scalar_values))
    predicted_target_numpy = np.array([x[0] for x in predicted_target_numpy])



    if not isinstance(data, np.ndarray):
        if hasattr(data, 'values'):
            data = data.values
        else:
            data = np.array(data)

    if folder_name is None:
        folder_name = "Training_Run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(folder_name, exist_ok=True)

    data = feature_1_denormalize(data, scaler_1 = scaler_1, scaler_2= scaler_2)


    plt.plot(data, label='This is Target data', color = 'red')
    plt.plot(predicted_target_numpy, label = 'This is the prediction', color ='blue')
    plt.legend()
    plt.show()

    # Step 8: Call your TPG plot saver function to save graphs
    run_TPG_link(
        predictions=predicted_target_numpy,
        actual=data,
        target="MPN5P",
        folder_name=folder_name
    )

    print(f"Training complete. Plots saved in folder: {folder_name}")
    print(f"The Testing Loss for this model over the testing data is : {testing_loss}")
    return model

def objective(trial, MLP_model_class, RFO_data ,search_space, scaler_1, scaler_2):
    print(f"We are inside the Objective Function")

    # your hyperparameter suggestions
    lr = trial.suggest_float("lr", *search_space['lr'], log=True)
    l2_lambda = trial.suggest_float("l2_lambda", *search_space['l2_lambda'])
    input_size = trial.suggest_categorical("input_size", search_space['input_size'])
    hidden_size = trial.suggest_categorical("hidden_size", search_space['hidden_size'])
    batch_size = trial.suggest_categorical("batch_size", search_space['batch_size'])
    alpha = trial.suggest_float("alpha", *search_space['alpha'])

    model = MLP_model_class(hidden_size=hidden_size, input_feature=input_size)

    val_losses = []
    train_val_pairs = rolling_forecast_origin_split(RFO_data)

    for train_data, val_data in train_val_pairs:

        model, _, _, _ = train_model_loop(
            model=model, alpha=alpha, epochs=10, lr=lr,
            training_data=train_data, batch_size=batch_size,
            l2_lambda=l2_lambda, window_size=input_size,
            scaler_1= scaler_1, scaler_2=scaler_2
        )
        val_loss, _ = evaluate(model, val_data=val_data, batch_size=batch_size, window_size=input_size, scaler_1 = scaler_1, scaler_2 = scaler_2)
        val_losses.append(val_loss)

    avg_val_loss = sum(val_losses) / len(val_losses)
    return avg_val_loss

def run_optuna_search(MLP_model_class, n_trials=25, folder_name=None, search_space=None):

    # Data loading
    RFO_data, test_data, scaler_1, scaler_2 = dataloader()
    data = np.concatenate((RFO_data, test_data), axis=0)

    # Define objective wrapper to pass model class and RFO_data to objective
    print(f"Troubleshooting befire optuna_objective")
    def optuna_objective(trial):
        print(f"We are before Objective Function")
        return objective(trial, MLP_model_class=MLP_model_class, RFO_data=RFO_data, search_space=search_space, scaler_1=scaler_1, scaler_2 = scaler_2)

    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_objective, n_trials=n_trials)

    best_params = study.best_trial.params
    print("Best Hyperparameters:", best_params)

    # Train best model on full RFO_data
    model = MLP_model_class(hidden_size=best_params['hidden_size'], input_feature=best_params['input_size'])

    model, _, _, _ = train_model_loop(
        model=model,
        alpha=best_params['alpha'],
        epochs=40,
        lr=best_params['lr'],
        training_data=RFO_data,
        batch_size=best_params['batch_size'],
        l2_lambda=best_params['l2_lambda'],
        window_size=best_params['input_size'],
        scaler_1= scaler_1, scaler_2 = scaler_2
    )

    # Final evaluation on test set
    test_loss, test_predictions = evaluate(
        model=model,
        val_data=data,
        batch_size=best_params['batch_size'],
        window_size=best_params['input_size'],
        scaler_1= scaler_1,
        scaler_2 = scaler_2
    )

    # Process predictions (your old code)
    predicted_target_numpy = []
    for phase_list in test_predictions:
        scalar_values = [t.item() for t in phase_list]
        np_array = np.array(scalar_values)
        predicted_target_numpy.append(np_array)
    predicted_target_numpy = np.array([x[0] for x in predicted_target_numpy])

    actual_data = feature_1_denormalize(data, scaler_1=scaler_1, scaler_2=scaler_2)
    predicted_target_numpy = feature_1_denormalize(predicted_target_numpy, scaler_1=scaler_1, scaler_2=scaler_2)

    # if not isinstance(actual_data, np.ndarray):
    #     if hasattr(actual_data, 'values'):
    #         actual_data = actual_data.values
    #     else:
    #         actual_data = np.array(actual_data)


    folder_name = folder_name or "TPG_Results"
    run_TPG_link(predictions=predicted_target_numpy, actual=actual_data, target="MPN5P", folder_name=folder_name)

def evaluate(model, val_data, batch_size, window_size, scaler_1, scaler_2):

    model.eval()
    with torch.no_grad():

        loss = 0
        prediction = []
        criterion = nn.MSELoss()

        batch_window_tuple_list = batch_window_generator(val_data, batch_size=batch_size, window_size=window_size)
        for batch_window_tuple in batch_window_tuple_list:
            batch, window = batch_window_tuple
            batch_prediction = []

            if len(batch) == 0:
                continue

            batch = list(batch)  # Ensure indexable in case it's a NumPy array

            for i in range(len(batch)):
                input_window = move_on(batch, window, i)
                input_window = input_window.unsqueeze(0)
                pred = model(input_window)
                batch_prediction.append(pred)

            batch_prediction_denormalized = feature_1_denormalize(batch_prediction, scaler_1 = scaler_1, scaler_2= scaler_2)
            batch_targets_denormalized = feature_1_denormalize(batch, scaler_1 = scaler_1, scaler_2 = scaler_2)


            # The batch_prediction_denormalized is a list of  numpy.float32
            # The batch_targets_denormalized is a list of  float64

            # Convert batch_prediction_denormalized list of numpy.float32 to a PyTorch tensor float32
            batch_prediction_tensor = torch.tensor(batch_prediction_denormalized, dtype=torch.float32)

            # Convert batch_targets_denormalized list of Python float64 to a PyTorch tensor float32
            batch_targets_tensor = torch.tensor(batch_targets_denormalized, dtype=torch.float64).to(torch.float32)

            # Optional: reshape if needed (for example, [batch_size, 1])
            batch_prediction_tensor = batch_prediction_tensor.reshape(-1, 1)
            batch_targets_tensor = batch_targets_tensor.reshape(-1, 1)

            # loss calculation ; l2 + base
            batch_base_loss = criterion(batch_prediction_tensor, batch_targets_tensor)

            #Recording
            loss += batch_base_loss.item()
            batch_np = np.array(batch_prediction_denormalized).reshape(-1, 1)
            for val in batch_np:
                prediction.append([val[0]])
    return loss, prediction

