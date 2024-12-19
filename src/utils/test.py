from .data import get_test_dataloader
from utils.pytorch import inference_dnn
from metrics import UnimodalPercentage, MeanAbsoluteErrorCustom, OrdinalContactSurface
from torchmetrics import Dice, JaccardIndex, ConfusionMatrix
from tqdm import tqdm
from datasets import Cityscapes_DP
from sys import stderr
from models.model_output import ModelOutput

def run_test(model, criterion, dataset_provider, TEST_ARGS):
    if isinstance(dataset_provider, Cityscapes_DP):
        print("> USING VALIDATION DATASET DUE TO CITYSCAPES\n", file=stderr)
        test_loader = get_test_dataloader(TEST_ARGS, dataset_provider.create_val())
    else:
        print("> USING TEST DATASET\n", file=stderr)
        test_loader = get_test_dataloader(TEST_ARGS, dataset_provider.create_test())
    
    model.eval()

    K = dataset_provider.get_num_classes()

    metrics = {
        'percentage_of_unimodal_px': UnimodalPercentage(dataset_provider.get_ordinality_tree()),
        'contact_surface': OrdinalContactSurface(dataset_provider.get_ordinality_tree()),
        'dice_coefficient_macro': Dice(average='macro', num_classes=K),
        'dice_coefficient_micro': Dice(average='micro', num_classes=K),
        'iou_macro': JaccardIndex(average='macro', num_classes=K, task='multiclass'),
        'iou_micro': JaccardIndex(average='micro', num_classes=K, task='multiclass'),
        'mae': MeanAbsoluteErrorCustom(num_classes=K),
        'confusion_matrix': ConfusionMatrix(task='multiclass', num_classes=K, normalize='true')
    }

    metrics = { name: metric.to(TEST_ARGS.device) for name, metric in metrics.items() }

    for data, target in tqdm(test_loader):  # iterate over test data
        output: ModelOutput = inference_dnn(model, data, TEST_ARGS)
        target = target.to(TEST_ARGS.device)
        
        probs = output.get_probs(criterion)
        for metric in metrics.values():
            metric.update(probs, target)

    def transform_metric_result(result):
        if type(result) == float:
            return round(result, 4)
        if len(result.shape) == 0:
            return round(float(result), 4)
        if len(result.shape) != 2:
            print(f"Invalid metric result shape {result.shape}", file=stderr)
            return
        result = result.tolist()
        for i, line in enumerate(result):
            for j, n in enumerate(line):
                result[i][j] = round(float(n), 4)
        return result

    metrics = { name: transform_metric_result(metric.compute()) for name, metric in metrics.items() }

    print("\n" + 20 * "-")
    print("TEST RESULTS\n")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print(20 * "-")

    return metrics
