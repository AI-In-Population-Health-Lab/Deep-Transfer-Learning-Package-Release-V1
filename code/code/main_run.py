import random
import time
import warnings
import os
import argparse




def main(args: argparse.Namespace):

    target = "--target "+str(' '.join(args.target)) if  (args.target) else ''
    source = '--source '+str(' '.join(args.source)) if  (args.source) else ''
    seed = '--seed='+str(args.seed) if  (args.seed) else ''
    epoch = '--epoch=' +str(args.epoch) if (args.epoch) else ''
    
    print("\n--------Run Data-based Deep Transfer Learning model---------")  
    print('1.DANN--unsupervised \n')
    os.system('python dann_synthetic_noTargetLabel_noTargetVal_outputAUC.py '+target+' '+source+' '+seed+' '+epoch)
    print('\n2.DANN--supervised \n')
    os.system('python dann_synthetic_withTargetLabel_outputAUC.py '+target+' '+source+' '+seed+' '+epoch)
    

    print("\n-------- Run Baseline Model ---------")
    print("1. TRAIN SOURCE MODEL \n")
    os.system('python learnSourceModel.py '+source+' '+seed+' '+epoch)
    print("\n 2. TRAIN TARGET MODEL \n")
    os.system('python learnTargetModel.py '+target+' '+seed+' '+epoch)
    print("\n 3. TRAIN combined MODEL \n")
    os.system('python learnCombineModel.py '+target+' '+source+' '+seed+' '+epoch)


    print("\n-------- Run Model-based Deep Transfer Learning model ---------")
    print(" 1. Fine-Tuning Last 1 Layer \n")
    os.system('python model-based-TL-TuneLast1Layer.py  '+target+' '+source+' '+seed+' '+epoch)
    print("\n 2. Fine-Tuning Last 2 Layers \n")
    os.system('python model-based-TL-TuneLast2Layers.py  '+target+' '+source+' '+seed+' '+epoch)
    print("\n 3. Fine-Tuning all Layers \n")
    os.system('python model-based-TL-TuneAllLayers.py  '+target+' '+source+' '+seed+' '+epoch)
    print('\n')
    






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DTL')
    parser.add_argument('--seed', default=1024, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--source', '--source_path', default='',type=str, nargs='+',
                        help='path of source data',dest='source')
    parser.add_argument('--target', '--target_path', default='',type=str, nargs='+',
                        help='path of target data',dest='target')
    parser.add_argument('--epoch', default=10, type=int, metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()

    print(args)

    main(args)
