import torch, json, os
os.chdir("..")

from torch.optim import Adam
from time import time as get_time
from utils.files import get_train_file_names
from utils.args import get_train_args, get_args_dict
from utils.data import get_train_val_dataloaders, get_train_dataloader
from utils.pytorch import setup_pytorch, finalize_pytorch, train_epoch_dnn, test_dnn
from sklearn.model_selection import KFold

TRAIN_ARGS, dataset_provider, model_provider, loss_provider = get_train_args()

model_folder, model_file, info_file = get_train_file_names(TRAIN_ARGS.model_name)

setup_pytorch(TRAIN_ARGS, model_folder)

best_valid_loss = 9999.
fold_losses = { k: { 'loss': best_valid_loss, 'n_epochs': 0 } for k in range(TRAIN_ARGS.k) }

train_dataset = dataset_provider.create_train()
val_dataset = dataset_provider.create_val()

unsupervised_dataset = None
if TRAIN_ARGS.semisupervised:
    unsupervised_dataset = dataset_provider.create_unsupervised()

def write_training_data(duration: float):
    with open(info_file, "w") as f:
        info = get_args_dict(TRAIN_ARGS)
        additional_info = {
            'folds': { k: fold_losses[k] for k in fold_losses.keys() if fold_losses[k]['loss'] != 9999. },
            'chosen_loss': best_valid_loss,
            'training_duration_per_fold_s': duration 
        }
        json.dump({ **info, **additional_info }, f, indent=2, sort_keys=True)

if TRAIN_ARGS.k == 1:
    folds = [(list(range(len(train_dataset))), list(range(len(val_dataset))))]
else:
    kfold = KFold(n_splits=TRAIN_ARGS.k)
    folds = kfold.split(train_dataset)

for fold, (train_idx, val_idx) in enumerate(folds):
    cur_patience = TRAIN_ARGS.patience
    cur_refinement = TRAIN_ARGS.refinement
    cur_lr = TRAIN_ARGS.lr

    start_t = get_time()

    network = model_provider.create_model()
    model = network.to(TRAIN_ARGS.device)
    optimizer = Adam(model.parameters(), lr=TRAIN_ARGS.lr, weight_decay=1e-3)
    criterion = loss_provider.create_loss()

    train_loader, valid_loader = get_train_val_dataloaders(TRAIN_ARGS, train_dataset, val_dataset, train_idx, val_idx)
    unsupervised_train_loader = None if unsupervised_dataset is None else get_train_dataloader(TRAIN_ARGS, unsupervised_dataset)

    print("\n" + 30 * "-")
    print(f"Fold {fold + 1}")
    fold_model_file = model_file + f".fold{fold+1}"

    total_epochs = 0
    for epoch in range(1, TRAIN_ARGS.max_epochs+1):
        total_epochs += 1
        print()
        print(f"> Fold {fold + 1} | Epoch {epoch}/{TRAIN_ARGS.max_epochs}, cur_patience = {cur_patience}, cur_refinement = {cur_refinement}, lr = {cur_lr}")
        
        train_epoch_dnn(fold+1, epoch+1, model, train_loader, unsupervised_train_loader, optimizer, criterion, TRAIN_ARGS)

        valid_loss = test_dnn(fold+1, epoch+1, model, valid_loader, criterion, TRAIN_ARGS)

        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), model_file)            
            best_valid_loss = valid_loss
            print("Valid loss is the best valid loss in every fold!")

        if valid_loss < fold_losses[fold]['loss']:
            torch.save(model.state_dict(), fold_model_file)
            fold_losses[fold]['loss'] = valid_loss
            fold_losses[fold]['n_epochs'] = total_epochs
            cur_patience = TRAIN_ARGS.patience
            print("Valid loss improved!")
        else:
            if cur_patience <= 0:
                if cur_refinement <= 0:
                    print("Stopping training due to exhausted patience and refinement!")
                    break
                else:
                    print("Exhausted patience, starting refinement round.")
                    cur_refinement -= 1 # decrement refinement
                    cur_patience = TRAIN_ARGS.patience # reset patience

                    # update lr
                    cur_lr *= TRAIN_ARGS.lr_fac
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = cur_lr

                    # load the best model to date
                    model.load_state_dict(torch.load(fold_model_file, map_location=TRAIN_ARGS.device))
            else:
                print("Valid loss didn't improve, decreasing patience.")
                cur_patience -= 1

    write_training_data(get_time()-start_t)
    if TRAIN_ARGS.only_first_fold:# and fold > 0: # temporarily training 2 folds
        break


write_training_data(get_time()-start_t)
finalize_pytorch()

print('\nTraining took: {:.2f}s for {} epochs'.format(get_time()-start_t, epoch))
print(f'> Model: {os.path.join(*model_folder.split("/")[-2:])} (loss = {best_valid_loss})')
folds = ", ".join([ f"{k}: {v}" for k, v in fold_losses.items() ])
print(f'Folds: {folds}')
