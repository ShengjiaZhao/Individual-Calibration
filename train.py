import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from crime_dataset import *
from models import *
from utils import *
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=int, default=0)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--dataset', type=str, default='crime')
parser.add_argument('--coeff', type=float, default=0.1)    # When coeff is 0 completely PAI loss, when 1 completely NLL loss
parser.add_argument('--recalibrate', action='store_true')
parser.add_argument('--group_index', type=int, default=-1) # If group_index >= 0, then apply group recalibration for the group_index-th feature. Otherwise apply global recalibration
args = parser.parse_args()
device = torch.device("cuda:%s" % args.gpu)
args.device = device

while True:   # Run as many times as possible to make plots with error bars
    dataset = CrimeDataset(device=device)

    # Create logging directory and create tensorboard and text loggers
    log_dir = 'log/coeff=%.2f-recalib=%r-%d-run=%d/' % (args.coeff, args.recalibrate, args.group_index, args.run)
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    logger = open(os.path.join(log_dir, 'result.txt'), 'w')

    # Define model
    model = FcSmall(x_dim=dataset.x_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Variables to determine early stopping
    test_losses = []

    for i in range(1, 50000):
        # Apply standard training step
        model.train()
        optimizer.zero_grad()
        
        train_x, train_y = dataset.train_batch(128)
        cdf, loss_cdf, loss_stddev, loss_nll = model.eval_all(train_x, train_y)
        loss = (1 - args.coeff) * loss_cdf + args.coeff * loss_nll

        loss.backward()
        optimizer.step()

        # Log training performance
        if i % 10 == 0:
            writer.add_scalar('cdf_l1', loss_cdf, global_step=i)
            writer.add_scalar('stddev', loss_stddev, global_step=i)
            writer.add_scalar('nll', loss_nll, global_step=i)
            writer.add_scalar('total', loss, global_step=i)

        # Log test performance
        if i % 100 == 0:
            # Computer test loss
            model.eval()
            with torch.no_grad():
                val_x, val_y = dataset.test_batch()
                cdf, loss_cdf, loss_stddev, loss_nll = model.eval_all(val_x, val_y)
                loss = (1 - args.coeff) * loss_cdf + args.coeff * loss_nll

            writer.add_histogram('cdf_test', cdf, global_step=i)
            writer.add_scalar('cdf_l1_test', loss_cdf, global_step=i)
            writer.add_scalar('stddev_test', loss_stddev, global_step=i)
            writer.add_scalar('nll_test', loss_nll, global_step=i)
            writer.add_scalar('total_test', loss, global_step=i)

            # Early stop if at least 3k iterations and test loss does not improve on average in the past 1k iteration
            test_losses.append(float(loss.cpu().numpy()))
            if len(test_losses) > 30 and np.mean(test_losses[-10:]) >= np.mean(test_losses[-20:-10]):
                # Log the final test loss
                logger.write("%d %f %f %f " % (i, loss, loss_nll, loss_stddev))

                # Compute the recalibration function if necessary
                if args.recalibrate:
                    train_x, train_y = dataset.train_batch()
                    model.recalibrate(train_x, train_y, args.group_index)
                break

            print("Iteration %d, test loss = %.3f" % (i, loss))

    # Evaluate stddev
    stddev_mc = eval_stddev(val_x, model)
    logger.write("%f\n" % stddev_mc)

    # Evaluate calibration of subsets
    ratios = np.linspace(0.01, 1.0, 50)
    for ratio in ratios:
        ece_s, ece_l = eval_ece(val_x, val_y, model, ratio)
        logger.write("%f " % max(ece_s, ece_l))
    logger.write('\n')

    # Evaluate calibration of identifiable subgroups
    if dataset.x_dim != 1:
        ece_w2, dim = eval_ece_by_2dim(val_x, val_y, model,
                                       plot_func=lambda image: writer.add_image('calibration-w2', image))
        logger.write("%f %s-%s-%d\n" % (ece_w2, dataset.names[dim[0]], dataset.names[dim[1]], dim[2]))
    logger.close()

    args.run += 1
