import random
import time
import warnings
import sys
import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torch.nn.functional as F

from collections import Counter
from feedforward import BackboneClassifierNN_M4
from tools.utils import AverageMeter, ProgressMeter, accuracy, ForeverDataIterator

from tools.lr_scheduler import StepwiseLR
from data_processing import  prepare_datasets_combineSourceTarget
import numpy
from sklearn import metrics

import pandas as pd

sys.path.append('.')




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

source_train_fold_loc = '../../data/synthetic_data_v2/source_train/'
target_train_fold_loc = '../../data/synthetic_data_v2/target_train/'
target_test_fold_loc = '../../data/synthetic_data_v2/target_test/'
results_fold_loc ='../../results/accuracy/'
learned_model_fold_loc ='../../results/learned_model/combine_model/'
prob_fold_loc ='../../results/auc/'

def main(args: argparse.Namespace):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # data loading code
    label_key = "diagnosis"
    validation_split = 0.125

    source_train_path = source_train_fold_loc + args.source_train_path + ".csv"
    target_train_path = target_train_fold_loc + args.target_train_path + ".csv"
    target_test_path = target_test_fold_loc + "findings_final_0814_seed-1494714102_size10000.csv"
    learned_model_path = learned_model_fold_loc  + args.source_train_path+"_"+args.target_train_path + "_" + str(args.seed) + ".pth"

    combine_train_dataset, target_val_dataset, target_test_dataset = prepare_datasets_combineSourceTarget(source_train_path,
                                                                                        target_train_path,
                                                                                        target_test_path, label_key,
                                                                                        validation_split)

    train_dataset = combine_train_dataset
    val_dataset = target_val_dataset
    test_dataset = target_test_dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_iter = ForeverDataIterator(train_loader)
    
    # create source model only by using source train and source validate data
    # input -> FC -> relu -> FC -> relu -> output num_classes
    # no softmax because that's handled by cross entropy loss already
    backbone_in_dim = train_dataset.features.shape[1]
    print("input dimension:", backbone_in_dim)
    num_classes = len(Counter(train_dataset.labels).keys())
    print("num_classes:")
    print(num_classes)


    classifier = BackboneClassifierNN_M4(backbone_in_dim, num_classes).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = StepwiseLR(optimizer, init_lr=args.lr, gamma=0.001, decay_rate=0.75)

    print("classifier:")
    print(classifier)
    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_iter, classifier, optimizer, lr_scheduler, epoch, args)
        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args)
        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:

            torch.save(classifier.state_dict(), learned_model_path) # newly added to save the best models
            best_acc1 = acc1
    if best_acc1==0 :
        torch.save(classifier.state_dict(), learned_model_path)

    print("best_validation_acc1 = {:3.1f}".format(best_acc1))
    # load source model
    print("load best model")
    classifier.load_state_dict(torch.load(learned_model_path))
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in classifier.state_dict():
        print(param_tensor, "\t", classifier.state_dict()[param_tensor].size())

    # evaluate on test set
    test_acc1, total_y_true, total_y_pred1, total_y_pred2, total_y_pred3, total_y_pred4, total_y_diagnosis, total_y_correct = getAUROC(
        test_loader, classifier, args)
    print("test_acc1 = {:3.1f}".format(test_acc1))
    csvFilename = prob_fold_loc + "source_model_prob_" + args.source_train_path + "_" + str(args.seed) + ".csv"
    avg_auc, auc_I, auc_M, auc_P, auc_R = printListToFile(csvFilename, total_y_true, total_y_pred1, total_y_pred2,
                                                          total_y_pred3, total_y_pred4, total_y_diagnosis,
                                                          total_y_correct)
    return test_acc1, avg_auc, auc_I, auc_M, auc_P, auc_R

def train(train_source_iter: ForeverDataIterator, model: nn.Module, optimizer: SGD, lr_scheduler: StepwiseLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i in range(args.iters_per_epoch):
        lr_scheduler.step()

        # measure data loading time
        data_time.update(time.time() - end)

        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # compute output
        y_s = model(x_s)
        loss = F.cross_entropy(y_s, labels_s)

        # update meters
        losses.update(loss.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate(val_loader: DataLoader, model: nn.Module, args: argparse.Namespace):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (features, target) in enumerate(val_loader):
            features = features.to(device)
            target = target.to(device)

            # compute output
            output = model(features)
            loss = F.cross_entropy(output, target)

            prob = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[:, 1]

            # measure accuracy and record loss
            acc1 = accuracy(output, target)
            losses.update(loss.item(), features.size(0))
            top1.update(acc1[0].item(), features.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))

    return top1.avg


def printListToFile(fileName, total_y_true, total_y_pred1, total_y_pred2, total_y_pred3, total_y_pred4,
                    total_y_diagnosis, total_y_correct):
    a = numpy.asarray(total_y_true).astype(int)
    b1 = numpy.asarray(total_y_pred1)
    b2 = numpy.asarray(total_y_pred2)
    b3 = numpy.asarray(total_y_pred3)
    b4 = numpy.asarray(total_y_pred4)
    c = numpy.asarray(total_y_diagnosis).astype(int)
    d = numpy.asarray(total_y_correct).astype(int)
    df = pd.DataFrame({"y_true": a, "p0": b1, "p1": b2, "p2": b3, "p3": b4, "prediction": c, "correct": d})
    df.to_csv(fileName, index=False)

    # calculate auc
    # I:0; M:1; P:2; R:3
    df['I_category'] = 'F'
    df.loc[df['y_true'] == 0, "I_category"] = "T"
    df['M_category'] = 'F'
    df.loc[df['y_true'] == 1, "M_category"] = "T"
    df['P_category'] = 'F'
    df.loc[df['y_true'] == 2, "P_category"] = "T"
    df['R_category'] = 'F'
    df.loc[df['y_true'] == 3, "R_category"] = "T"
    fpr, tpr, thresholds = metrics.roc_curve(df['I_category'], df['p0'], pos_label='T')
    auc_I = metrics.auc(fpr, tpr)
    fpr, tpr, thresholds = metrics.roc_curve(df['M_category'], df['p1'], pos_label='T')
    auc_M = metrics.auc(fpr, tpr)
    fpr, tpr, thresholds = metrics.roc_curve(df['P_category'], df['p2'], pos_label='T')
    auc_P = metrics.auc(fpr, tpr)
    fpr, tpr, thresholds = metrics.roc_curve(df['R_category'], df['p3'], pos_label='T')
    auc_R = metrics.auc(fpr, tpr)
    avg_auc = (auc_I + auc_M + auc_P + auc_R) / 4
    return avg_auc, auc_I, auc_M, auc_P, auc_R


def getAUROC(val_loader: DataLoader, model: nn.Module, args: argparse.Namespace):
    total_y_pred1 = numpy.array([[]])
    total_y_pred2 = numpy.array([[]])
    total_y_pred3 = numpy.array([[]])
    total_y_pred4 = numpy.array([[]])
    total_y_diagnosis = numpy.array([[]])
    total_y_correct = numpy.array([[]])
    total_y_true = numpy.array([])

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (features, target) in enumerate(val_loader):
            features = features.to(device)
            target = target.to(device)

            # compute output
            output = model(features)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)
            losses.update(loss.item(), features.size(0))
            top1.update(acc1[0].item(), features.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            probAll = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
            total_y_pred1 = numpy.append(total_y_pred1, probAll[:, 0])
            total_y_pred2 = numpy.append(total_y_pred2, probAll[:, 1])
            total_y_pred3 = numpy.append(total_y_pred3, probAll[:, 2])
            total_y_pred4 = numpy.append(total_y_pred4, probAll[:, 3])
            total_y_true = numpy.append(total_y_true, target.cpu().numpy())
            _, diagnosis = output.topk(1, 1, True, True)
            diagnosis = diagnosis.t()
            total_y_diagnosis = numpy.append(total_y_diagnosis, diagnosis.cpu().numpy())
            correct = diagnosis.eq(target.view(1, -1).expand_as(diagnosis))
            total_y_correct = numpy.append(total_y_correct, correct.cpu().numpy())

        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))

    return top1.avg, total_y_true, total_y_pred1, total_y_pred2, total_y_pred3, total_y_pred4, total_y_diagnosis, total_y_correct


if __name__ == '__main__':
    source_train_paths = ['findings_final_0814_seed1591536269_size10000',
                      'findings_final_0814-portion1ita06round14_seed2016863826_size10000']


    target_train_paths= ['findings_final_0814_seed-972126700_size500']
    seed_paths = [14942]


    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay',default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('-i', '--iters-per-epoch', default=313, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--seed', default=seed_paths, type=int,nargs='+',
                        help='seed for initializing training. ')
    parser.add_argument('--source', '--source_path', default=source_train_paths,  nargs='+',
                        help='path of source data',dest='source')
    parser.add_argument('--target', '--target_path', default=target_train_paths,  nargs='+',
                        help='path of target data',dest='target')

    args = parser.parse_args()
    #print(args)

    source_train_paths = args.source
    seed_paths = args.seed 
    target_train_paths = args.target

    

    d_kl_dict = {}
    d_kl_dict['findings_final_0814'] = 0
    d_kl_dict['findings_final_0814-portion1ita06round14'] = 1
    d_kl_dict['findings_final_0814-portion1ita13round20'] = 5
    d_kl_dict['findings_final_0814-portion1ita16round14'] = 10
    d_kl_dict['findings_final_0814-portion1ita27round9'] = 20
    d_kl_dict['findings_final_0814-portion1ita29round18'] = 30
   


    with open(results_fold_loc + "/combineModel_acc_auc.txt", "w") as f:
        f.write(f"d_kl,source_train_path,seed_index,seed,test_acc1,avg_auc,auc_I,auc_M,auc_P,auc_R\n")
        for i in range(len(target_train_paths)):
            args.target_train_path = target_train_paths[i]
            for i in range(len(source_train_paths)):
                args.source_train_path = source_train_paths[i]
                print(args.source_train_path)
                d_kl = -1
                for key in d_kl_dict.keys():
                    if key in args.source_train_path:
                        d_kl = d_kl_dict[key]
                for seed_index in range(len(seed_paths)):
                    args.seed = seed_paths[seed_index]
                    test_acc1, avg_auc,auc_I,auc_M,auc_P,auc_R = main(args)
                    f.write(
                        f"{d_kl},{args.source_train_path},{seed_index},{args.seed},{test_acc1},{avg_auc},{auc_I},{auc_M},{auc_P},{auc_R}\n")

