import random
import time
import warnings
import sys
import argparse
import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append('')
from dalib.adaptation.mcd import entropy, classifier_discrepancy
import dalib.vision.datasets as datasets
import dalib.vision.models as models
from tools.utils import AverageMeter, ProgressMeter, accuracy, create_exp_dir, ForeverDataIterator
from tools.transforms import ResizeImage

from data_processing import prepare_datasets, prepare_datasets_returnSourceVal
from feedforward import BackboneNN, ClassifierHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

source_train_fold_loc = '/Users/yeye/Documents/CMU_course_project/Transfer-Learning/syntheticData_Exp_11082021_code_result/data/synthetic_data_v2/source_train/'
target_train_fold_loc = '/Users/yeye/Documents/CMU_course_project/Transfer-Learning/syntheticData_Exp_11082021_code_result/data/synthetic_data_v2/target_train/'
target_test_fold_loc = '/Users/yeye/Documents/CMU_course_project/Transfer-Learning/syntheticData_Exp_11082021_code_result/data/synthetic_data_v2/target_test/'
results_fold_loc ='/Users/yeye/Documents/CMU_course_project/Transfer-Learning/syntheticData_Exp_11082021_code_result/results/accuracy/'
learned_model_fold_loc ='/Users/yeye/Documents/CMU_course_project/Transfer-Learning/syntheticData_Exp_11082021_code_result/results/learned_model/mcd/'


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

    #source_train_dataset, target_train_dataset, target_val_dataset, target_test_dataset = prepare_datasets(source_train_path, target_train_path, target_test_path, label_key, validation_split)
    source_train_dataset, source_val_dataset, target_test_dataset, target_train_dataset = prepare_datasets_returnSourceVal(
        source_train_path,
        target_train_path,
        target_test_path,
        label_key,
        validation_split)

    learned_G_model_path = learned_model_fold_loc + args.source_train_path + "-" + args.target_train_path + "_" + str(
        args.seed) + "-MCD-G.pth"
    learned_F1_model_path = learned_model_fold_loc + args.source_train_path + "-" + args.target_train_path + "_" + str(
        args.seed) + "-MCD-F1.pth"
    learned_F2_model_path = learned_model_fold_loc + args.source_train_path + "-" + args.target_train_path + "_" + str(
        args.seed) + "-MCD-F2.pth"

    train_source_loader = DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    #val_loader = DataLoader(target_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    val_loader = DataLoader(source_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    G_in_dim = source_train_dataset.features.shape[1]
    G = BackboneNN(G_in_dim)

    num_classes = 4
    # two classifier heads
    F1 = ClassifierHead(G.out_features, num_classes, args.bottleneck_dim).to(device)
    F2 = ClassifierHead(G.out_features, num_classes, args.bottleneck_dim).to(device)

    # define optimizer
    # the learning rate is fixed according to origin paper
    optimizer_g = SGD(G.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_f = SGD(F1.get_parameters()+F2.get_parameters(), momentum=0.9, lr=args.lr, weight_decay=0.0005)

    # start training
    best_acc1 = 0.
    best_results = None
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, G, F1, F2, optimizer_g, optimizer_f, epoch, args)

        # evaluate on validation set
        results = validate(val_loader, G, F1, F2, args)

        # remember best acc@1 and save checkpoint
        if max(results) > best_acc1:
            best_G, best_F1, best_F2 = copy.deepcopy(G.state_dict()), copy.deepcopy(F1.state_dict()), copy.deepcopy(
                F2.state_dict())
            torch.save(G.state_dict(), learned_G_model_path)  # newly added to save the best models
            torch.save(F1.state_dict(), learned_F1_model_path)  # newly added to save the best models
            torch.save(F2.state_dict(), learned_F2_model_path)  # newly added to save the best models
            best_acc1 = max(results)
            best_results = results

    print("best_acc1 = {:3.1f}, results = {}".format(best_acc1, best_results))
    if best_results.index(max(best_results)) == 0:
        best_classifier = 'F1'
    elif best_results.index(max(best_results)) == 1:
        best_classifier = 'F2'
    print("best_classifer:", best_classifier)

    # evaluate on test set
    G.load_state_dict(best_G)
    F1.load_state_dict(best_F1)
    F2.load_state_dict(best_F2)
    results = validate(test_loader, G, F1, F2, args)

    if best_classifier == 'F1':
        test_acc = results[0]
    elif best_classifier == 'F2':
        test_acc = results[1]
    print("test_acc1 = {:3.1f}".format(test_acc))
    return best_acc1, test_acc, best_classifier



def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          G: nn.Module, F1: ClassifierHead, F2: ClassifierHead,
          optimizer_g: SGD, optimizer_f: SGD, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    G.train()
    F1.train()
    F2.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        # measure data loading time
        data_time.update(time.time() - end)

        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)
        x = torch.cat((x_s, x_t), dim=0)
        assert x.requires_grad is False

        # Step A train all networks to minimize loss on source domain
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        g = G(x)
        y_1 = F1(g)
        y_2 = F2(g)
        y1_s, y1_t = y_1.chunk(2, dim=0)
        y2_s, y2_t = y_2.chunk(2, dim=0)

        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        loss = F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s) + \
               0.01 * (entropy(y1_t) + entropy(y2_t))
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        # Step B train classifier to maximize discrepancy
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        g = G(x)
        y_1 = F1(g)
        y_2 = F2(g)
        y1_s, y1_t = y_1.chunk(2, dim=0)
        y2_s, y2_t = y_2.chunk(2, dim=0)
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        loss = F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s) + \
               0.01 * (entropy(y1_t) + entropy(y2_t)) - classifier_discrepancy(y1_t, y2_t) * args.trade_off
        loss.backward()
        optimizer_f.step()

        # Step C train genrator to minimize discrepancy
        for k in range(args.num_k):
            optimizer_g.zero_grad()
            g = G(x)
            y_1 = F1(g)
            y_2 = F2(g)
            y1_s, y1_t = y_1.chunk(2, dim=0)
            y2_s, y2_t = y_2.chunk(2, dim=0)
            y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
            mcd_loss = classifier_discrepancy(y1_t, y2_t) * args.trade_off
            mcd_loss.backward()
            optimizer_g.step()

        cls_acc = accuracy(y1_s, labels_s)[0]
        tgt_acc = accuracy(y1_t, labels_t)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_t.size(0))
        trans_losses.update(mcd_loss.item(), x_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, G: nn.Module, F1: ClassifierHead,
             F2: ClassifierHead, args: argparse.Namespace) -> Tuple[float, float]:
    batch_time = AverageMeter('Time', ':6.3f')
    top1_1 = AverageMeter('Acc_1', ':6.2f')
    top1_2 = AverageMeter('Acc_2', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1_1, top1_2],
        prefix='Test: ')

    # switch to evaluate mode
    G.eval()
    F1.eval()
    F2.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            g = G(images)
            y1, y2 = F1(g), F2(g)

            # measure accuracy and record loss
            acc1, = accuracy(y1, target)
            acc2, = accuracy(y2, target)
            top1_1.update(acc1[0], images.size(0))
            top1_2.update(acc2[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc1 {top1_1.avg:.3f} Acc2 {top1_2.avg:.3f}'
              .format(top1_1=top1_1, top1_2=top1_2))

    return top1_1.avg, top1_2.avg


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--num-k', type=int, default=4, metavar='K',
                        help='how many steps to repeat the generator update')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('-i', '--iters-per-epoch', default=313, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--bottleneck-dim', default=128, type=int)
    parser.add_argument('--center-crop', default=False, action='store_true')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    args = parser.parse_args()


    source_train_paths = ['findings_final_0814_seed1591536269_size10000',
                          'findings_final_0814-portion1ita06round14_seed2016863826_size10000',
                          'findings_final_0814-portion1ita13round20_seed1708886178_size10000',
                          'findings_final_0814-portion1ita16round14_seed1948253030_size10000',
                          'findings_final_0814-portion1ita21round14_seed1879396416_size10000',
                          'findings_final_0814-portion1ita27round9_seed1940262766_size10000']

    target_train_paths = ['findings_final_0814_seed238506806_size1000',
                          'findings_final_0814_seed1033059257_size2000',
                          'findings_final_0814_seed678668699_size3000',
                          'findings_final_0814_seed-1872107095_size4000',
                          'findings_final_0814_seed-190708218_size5000',
                          'findings_final_0814_seed2132231585_size10000',
                          'findings_final_0814_seed-972126700_size500',
                          'findings_final_0814_seed-1331694080_size100']


    d_kl_dict = {}
    d_kl_dict['findings_final_0814'] = 0
    d_kl_dict['findings_final_0814-portion1ita06round14'] = 1
    d_kl_dict['findings_final_0814-portion1ita13round20'] = 5
    d_kl_dict['findings_final_0814-portion1ita16round14'] = 10
    d_kl_dict['findings_final_0814-portion1ita21round14'] = 15
    d_kl_dict['findings_final_0814-portion1ita27round9'] = 20

    seed_paths = [14942, 43277, 79280, 8463, 12650]

    assert (len(source_train_paths) == len(d_kl_dict.keys()))

    alpha = 1

    for j in range(len(target_train_paths)):
        args.target_train_path = target_train_paths[j]
        size = target_train_paths[j].split("size")[1]
        with open(results_fold_loc + "/mcd_complete_noTargetLabel_log_" + size + ".txt", "w") as f:
            f.write(f"d_kl,source_train_path,target_train_path,seed_index,seed,best_classifier,validate_acc,test_acc\n")
            for i in range(len(source_train_paths)):
                args.source_train_path = source_train_paths[i]
                print(args.source_train_path, args.target_train_path)
                # args.trade_off = alpha * (1 - np.exp(-d_kl[i] / (alpha)))
                # args.trade_off = d_kl[i]
                args.trade_off = 1.0
                d_kl = -1
                for key in d_kl_dict.keys():
                    if key in args.source_train_path:
                        d_kl = d_kl_dict[key]
                for seed_index in range(len(seed_paths)):
                    args.seed = seed_paths[seed_index]
                    validate_acc, test_acc, best_classifier = main(args)
                    f.write(
                        f"{d_kl},{args.source_train_path},{args.target_train_path},{seed_index},{args.seed},{best_classifier},{validate_acc},{test_acc},\n")
                    print(
                        f"{d_kl},{args.source_train_path},{args.target_train_path},{seed_index},{args.seed},{best_classifier},{validate_acc},{test_acc}\n")


