import argparse
import pytorch_lightning as pl
import os
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.dataset import plNBUDataset
from src.model.PreMixHuge import PreMixHuge
from src.util import check_and_make


def get_args_parser():
    parser = argparse.ArgumentParser('Training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--ms_chans', default=8, type=int)
    parser.add_argument('--embed_dim', default=32, type=int)
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--pf_kernel', default=3, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument(
        '--activation',
        choices=[
            "sigmoid",
            "tanh+relu",
            "tanh+elu",
            "softsign+elu",
            "softsign+relu"],
        type=str,
        help='activation function')
    parser.add_argument('--beta', default=None, type=float)
    parser.add_argument('--EWFM', action='store_true')
    parser.add_argument('--rgb_c', default='2,1,0')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--sensor', default='wv2', type=str)
    parser.add_argument('--test_freq', default=10, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    model_name = "PreMixHuge"
    output_dir = (
        f"log_m={model_name}_s={args.sensor}_l={args.num_layers}_d={args.embed_dim}"
        f"_k={args.kernel_size}_pfk={args.pf_kernel}_EWFM={args.EWFM}"
        f"_b={args.beta}_a={args.activation}"
    )
    check_and_make(output_dir)
    seed_everything(args.seed)

    dataset = plNBUDataset(args.data_dir,
                           args.batch_size,
                           args.num_workers,
                           args.pin_mem,
                           )

    model = PreMixHuge(lr=args.lr,
                       epochs=args.epochs,
                       bands=args.ms_chans,
                       rgb_c=args.rgb_c,
                       sensor=args.sensor,
                       embed_dim=args.embed_dim,
                       kernel_size=args.kernel_size,
                       pf_kernel=args.pf_kernel,
                       enable_EWFM=args.EWFM,
                       num_layers=args.num_layers,
                       beta=args.beta,
                       act=args.activation,
                       )

    wandb_logger = CSVLogger(name=output_dir, save_dir=output_dir)

    model_checkpoint = ModelCheckpoint(dirpath=output_dir,
                                       monitor='val/PSNR_mean',
                                       mode="max",
                                       save_top_k=1,
                                       auto_insert_metric_name=False,
                                       filename='ep={epoch}_PSNR={val/PSNR_mean:.4f}',
                                       every_n_epochs=args.test_freq
                                       )

    trainer = pl.Trainer(max_epochs=args.epochs,
                         accelerator="gpu",
                         devices=[args.device],
                         logger=wandb_logger,
                         check_val_every_n_epoch=args.test_freq,
                         callbacks=[model_checkpoint],
                         )

    trainer.fit(model, dataset)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

