from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
from datetime import datetime
from models import Fusion_Net

cuda_id = 0
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
parser = argparse.ArgumentParser(description="BIDualFusion for DTI prediction")
parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='BindingDB')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task",
                    choices=['random', 'random1', 'random2', 'random3', 'random4', 'cold', 'cold1', 'unseen_drug','unseen_target'])
parser.add_argument('--amp', action='store_true', help='Activate AMP (Automatic Mixed Precision) training')
parser.add_argument('--output_dir', type=str, metavar='DIR', help='output directory', default='random3')
parser.add_argument('--use_precomputed', action='store_true', help='Use precomputed features (ChemBERTa + ESM2)')
parser.add_argument('--precomputed_dir', type=str, default=None, help='Directory containing precomputed features')
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    set_seed(cfg.SOLVER.SEED)
    mkdir(cfg.RESULT.OUTPUT_DIR + f'{args.data}/{args.split}')

    print("start...")
    print(f"dataset:{args.data}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = os.path.join(f'../datasets', args.data, args.split)

    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, "val.csv")
    test_path = os.path.join(dataFolder, "test.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    train_precomputed_dir = None
    val_precomputed_dir = None
    test_precomputed_dir = None

    if args.use_precomputed:
        if args.precomputed_dir is None:
            args.precomputed_dir = os.path.join('..', 'datasets', args.data, args.split)

        train_precomputed_dir = os.path.join(args.precomputed_dir, 'train')
        val_precomputed_dir = os.path.join(args.precomputed_dir, 'val')
        test_precomputed_dir = os.path.join(args.precomputed_dir, 'test')

    train_dataset = DTIDataset(df_train.index.values, df_train, precomputed_features_dir=train_precomputed_dir)
    print(f'train_dataset:{len(train_dataset)}')
    val_dataset = DTIDataset(df_val.index.values, df_val, precomputed_features_dir=val_precomputed_dir)
    test_dataset = DTIDataset(df_test.index.values, df_test, precomputed_features_dir=test_precomputed_dir)

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}

    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    model = Fusion_Net(device=device, use_precomputed_features=args.use_precomputed, **cfg).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    torch.backends.cudnn.benchmark = True

    if args.amp:
        print("Activate AMP (Automatic Mixed Precision) training")

    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator,
                      args.data, args.split, use_amp=args.amp, **cfg)
    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/{args.split}/model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/{args.split}/config.txt"), "w") as wf:
        wf.write(str(dict(cfg)))

    print(f"\nDirectory for saving result: {cfg.RESULT.OUTPUT_DIR}{args.data}/{args.output_dir}")
    print(f'\nend...')

    return result


if __name__ == '__main__':
    torch.cuda.empty_cache()
    print(f"start time: {datetime.now()}")
    s = time()
    result = main()
    e = time()
    print(f"end time: {datetime.now()}")
    print(f"Total running time: {round(e - s, 2)}s, ")
