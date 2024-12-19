import argparse, os, json, hashlib
from datasets import DatasetProvider
from models import ModelProvider
from losses import LossProvider
from utils import custom_import as imp
from typing import Any, Tuple
from activations import ActivationProvider


# Helper class for neural network hyper-parameters
class Args:
    pass


def get_args_dict(args: Args):
    return { k: v for (k, v) in args.__dict__.items() if k != 'device' }


def print_args(args):
    print("\n" + 40 * "-")
    for arg in vars(args):
        print(arg, (20-len(arg)) * " ", getattr(args, arg))
    print(40 * "-" + "\n")


def get_train_args() -> Tuple[Any, DatasetProvider, ModelProvider]:
    parser = argparse.ArgumentParser(
        prog = 'PyTorch train',
        description = 'Training process of a DNN',
    )

    parser.add_argument('--model', required=True, choices=imp.get_possible_classes('model'), help="name of the model class to be used")
    parser.add_argument('--pretrained', action="store_true", help="whether to use a pretrained model")
    parser.add_argument('--only_first_fold', action="store_true", help="whether to only train the first fold (faster experimentation)")
    parser.add_argument('--dataset', required=True, choices=imp.get_possible_classes('dataset'), help="name of the dataset class to be used")
    parser.add_argument('--dataset_scale', required=True, type=float, help="scale of the dataset to use (1 = whole)")
    parser.add_argument('--dataset_mask_type', required=True, help="dataset mask type to use, dependant on dataset")
    parser.add_argument('--activation', required=True, choices=[*imp.get_possible_classes('activation'), 'None'], help="name of the loss class to be used")
    parser.add_argument('--loss', required=True, choices=imp.get_possible_classes('loss'), help="name of the loss class to be used")
    parser.add_argument('--batchsize', required=True, type=int, help="dataloader batch size")
    parser.add_argument('--lr', required=True, type=float, help="initial learning rate")
    parser.add_argument('--maxepochs', required=True, type=int, help="maximum number of training epochs")
    parser.add_argument('--patience', required=True, type=int, help="patience (help TBD)")
    parser.add_argument('--refinement', required=True, type=int, help="refinement (help TBD)")
    parser.add_argument('--lrfactor', required=True, type=float, help="learning rate refinement factor")
    parser.add_argument('--loginterval', default=5, type=int, help="interval between logs (seconds)")
    parser.add_argument('--kfolds', required=True, type=int, help="k value for k-folds cross validation")
    parser.add_argument('--regularization_weight', required=False, default=1., type=float, help="lambda for regularization (L = seg_loss + lambda * reg_loss)")
    parser.add_argument('--folder', required=False, type=str, default=".", help="folder for model saving")
    parser.add_argument('--nocuda', action="store_true", help="whether to use cuda")
    parser.add_argument('--semisupervised', action="store_true", help="whether to do semi supervised learning (dataset and loss must support it)")

    args = parser.parse_args()
    print_args(args)

    dataset_provider: DatasetProvider = imp.custom_import_class(args.dataset, 'dataset')(args.dataset_scale, args.dataset_mask_type)
    activation_provider: ActivationProvider = None if args.activation == 'None' else imp.custom_import_class(args.activation, 'activation')(num_classes=dataset_provider.get_num_classes(), tree=dataset_provider.get_ordinality_tree().dict_tree)
    activation = None if activation_provider is None else activation_provider.create_activation()
    model_outputs = dataset_provider.get_num_classes() if activation is None else activation.how_many_outputs()
    model_provider: ModelProvider = imp.custom_import_class(args.model, 'model')(pretrained=args.pretrained, n_channels=dataset_provider.get_num_channels(), how_many_outputs=model_outputs, activation=activation)
    loss_provider: LossProvider = imp.custom_import_class(args.loss, 'loss')(num_classes=dataset_provider.get_num_classes(), reg_weight=args.regularization_weight, tree=dataset_provider.get_ordinality_tree())

    TRAIN_ARGS = Args()

    # general params
    TRAIN_ARGS.use_cuda = not args.nocuda
    TRAIN_ARGS.seed = 1

    # architecture setup
    TRAIN_ARGS.batch_size = args.batchsize
    TRAIN_ARGS.num_classes = dataset_provider.get_num_classes()
    TRAIN_ARGS.dataset = args.dataset
    TRAIN_ARGS.activation = args.activation
    TRAIN_ARGS.model = args.model
    TRAIN_ARGS.pretrained = args.pretrained
    TRAIN_ARGS.loss = args.loss
    TRAIN_ARGS.k = args.kfolds
    TRAIN_ARGS.dataset_scale = args.dataset_scale
    TRAIN_ARGS.dataset_mask_type = args.dataset_mask_type
    TRAIN_ARGS.regularization_weight = args.regularization_weight
    TRAIN_ARGS.semisupervised = args.semisupervised

    # optimizer parameters
    TRAIN_ARGS.lr = args.lr

    # training protocol
    TRAIN_ARGS.only_first_fold = args.only_first_fold
    TRAIN_ARGS.max_epochs = args.maxepochs
    TRAIN_ARGS.patience = args.patience
    TRAIN_ARGS.refinement = args.refinement  # restart training after patience runs out with the best model, decrease lr by...
    TRAIN_ARGS.lr_fac = args.lrfactor    # ... the learning rate factor lr_fac: lr_new = lr_old*lr_fac
    TRAIN_ARGS.log_interval = args.loginterval  # seconds

    hash_seed = json.dumps(get_args_dict(TRAIN_ARGS)).encode("UTF-8")
    args_hash = hashlib.sha1(hash_seed).hexdigest()[:10]

    ssl = "" if not args.semisupervised else "_SSL"
    TRAIN_ARGS.model_name = f"{args.model}{ssl}_{args.dataset}{args.dataset_scale}_{args.dataset_mask_type}_{args.activation}_{args.loss}_{args.regularization_weight}_{args.maxepochs}e_{args_hash}_"
    TRAIN_ARGS.model_name = os.path.join(args.folder, TRAIN_ARGS.model_name)

    return TRAIN_ARGS, dataset_provider, model_provider, loss_provider


def get_test_args(args: Any = None) -> Tuple[Any, DatasetProvider, ModelProvider]:
    if args is None:
        parser = argparse.ArgumentParser(
            prog = 'PyTorch test',
            description = 'Testing process of a DNN',
        )

        parser.add_argument('modelname', help="trained model file name")
        # parser.add_argument('-m', '--model', default="DeepLabV3", choices=imp.get_possible_models(), help="name of the model manager class to be used")
        parser.add_argument('--dataset', required=False, choices=imp.get_possible_classes('dataset'), help="name of the dataset class to be used")
        parser.add_argument('--loss', required=False, choices=imp.get_possible_classes('loss'), help="name of the loss class to be used")
        parser.add_argument('--batchsize', default=8, type=int, help="dataloader batch size")
        parser.add_argument('--loginterval', default=5, type=int, help="interval between logs (seconds)")

        args, _ = parser.parse_known_args()
        print_args(args)

    model_folder = os.path.join("results", args.modelname) if "results/" not in args.modelname else args.modelname
    model_info_path = os.path.join(model_folder, "train.json")

    with open(model_info_path, "r") as f:
        TRAIN_ARGS = json.load(f)

    dataset = TRAIN_ARGS['dataset'] if args.dataset is None else args.dataset
    loss = TRAIN_ARGS['loss'] if args.loss is None else args.loss
    print(f"Using '{dataset}' dataset, '{TRAIN_ARGS['activation']}' activation, and '{loss}' loss.")

    dataset_provider: DatasetProvider = imp.custom_import_class(dataset, 'dataset')(TRAIN_ARGS['dataset_scale'], TRAIN_ARGS['dataset_mask_type'])
    activation_provider: ActivationProvider = None if TRAIN_ARGS['activation'] == 'None' else imp.custom_import_class(TRAIN_ARGS['activation'], 'activation')(num_classes=dataset_provider.get_num_classes(), tree=dataset_provider.get_ordinality_tree().dict_tree)
    activation = None if activation_provider is None else activation_provider.create_activation()
    model_outputs = dataset_provider.get_num_classes() if activation is None else activation.how_many_outputs()
    model_provider: ModelProvider = imp.custom_import_class(TRAIN_ARGS['model'], 'model')(pretrained=False, n_channels=dataset_provider.get_num_channels(), how_many_outputs=model_outputs, activation=activation)
    loss_provider: LossProvider = imp.custom_import_class(loss, 'loss')(num_classes=dataset_provider.get_num_classes(), reg_weight=TRAIN_ARGS['regularization_weight'] if 'regularization_weight' in TRAIN_ARGS else 1., tree=dataset_provider.get_ordinality_tree())

    TEST_ARGS = Args()
    TEST_ARGS.model_name = args.modelname
    TEST_ARGS.model_folder = model_folder

    TEST_ARGS.use_cuda = True
    TEST_ARGS.seed = 1

    TEST_ARGS.batch_size = args.batchsize
    TEST_ARGS.num_classes = dataset_provider.get_num_classes()

    TEST_ARGS.log_interval = args.loginterval  # seconds

    return TEST_ARGS, TRAIN_ARGS, dataset_provider, model_provider, loss_provider, model_folder, args.dataset is not None
