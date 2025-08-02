from Code.engine.MLP import SimpleBaseLine
from Code.engine.preprocessing import dataloader, feature_1_denormalize
from Code.engine.mlp_trainer import train_model_loop, evaluate
import matplotlib.pyplot as plt
import numpy as np



train_data, test_data , scaler_1, scaler_2 = dataloader()

print(f"train data has length oid {len(train_data)} ")
print(f"Some values are : {test_data[:5]}")