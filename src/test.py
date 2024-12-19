#!/bin/python3
import torch, os, json, sys
os.chdir("..")

from utils.args import get_test_args
from utils.pytorch import setup_pytorch, finalize_pytorch
from utils.test import run_test
from io import StringIO

TEST_ARGS, TRAIN_ARGS, dataset_provider, model_provider, loss_provider, model_folder, custom_dataset = get_test_args()


setup_pytorch(TEST_ARGS, model_folder)

results = {}

for fold in TRAIN_ARGS["folds"].keys():
    network = model_provider.create_model()
    model_path = os.path.join(TEST_ARGS.model_folder, f"best.model.fold{int(fold)+1}")
    print(model_path)
    if not os.path.exists(model_path):
        print('Model file not found, unable to load...')
        exit()

    # Load model and run test

    model = network.to(TEST_ARGS.device)
    model.load_state_dict(torch.load(model_path, map_location=TEST_ARGS.device))
    criterion = loss_provider.create_loss()
    print(10*"*" + f"\n Fold {int(fold)+1}")
    print(f"Model file loaded: {TEST_ARGS.model_name}")

    old_stdout = sys.stdout
    sys.stdout = my_stdout = StringIO()

    results[fold] = run_test(model, criterion, dataset_provider, TEST_ARGS)

    sys.stdout = old_stdout
    print(my_stdout.getvalue())
    results[fold]['stdout'] = my_stdout.getvalue()

suffix = ""
if custom_dataset:
    name = dataset_provider.__class__.__name__.removesuffix("_DP")
    suffix = f"_{name}"

with open(os.path.join(model_folder, f"test{suffix}.json"), "w") as f:
    json.dump(results, f, indent=2, sort_keys=True)

finalize_pytorch()
