import argparse
import os
import utils
import time
import torch
from torch import nn
from torch import optim

import numpy as np

from network import MobileNetV3_Large
import dataSet
from  modelsize import SizeEstimator


def get_argument():
    parser=argparse.ArgumentParser("Pytorch for MobileNet V3")
    parser.add_argument("--data_path",type=str,default="./cifar-10",help="The Path of train dataSet")
    parser.add_argument("--data_model", type=str, default="Cifar10", help="Example:ImageNet,Cifar10")

    parser.add_argument("--log_path",type=str,default="",help="The Path of train log")
    parser.add_argument("--model_path",type=str,default="logger/",help="The Path to Save trained model")
    parser.add_argument("--batch_size",type=int,default=32,help="The Batch Size of train data")
    parser.add_argument("--iter_epoch",type=int,default=800,help="num epoch of iteration")
    return parser.parse_args()

def train(train_loader,model,optimizer,criterion):
    # training
    total_loss = utils.AverageMeter()
    train_top1_accuracy = utils.AverageMeter()
    train_top5_accuracy = utils.AverageMeter()
    model.train()

    for step, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        ret = utils.accuracy(logits, targets, topk=(1, 5))
        n = inputs.size(0)
        total_loss.update(loss.item(), n)
        prec1 = ret[0]
        prec5 = ret[1]
        train_top1_accuracy.update(prec1.item(), n)
        train_top5_accuracy.update(prec5.item(), n)
        if step % 20 == 0:
            print("Train %d %e %f %f" % (
            step, total_loss.average, train_top1_accuracy.average, train_top5_accuracy.average))
    return total_loss,train_top1_accuracy.average,train_top5_accuracy.average

def valid(val_loader,model,criterion):
    # valid
    total_loss = utils.AverageMeter()
    val_top1_accuracy = utils.AverageMeter()
    val_top5_accuracy = utils.AverageMeter()
    model.eval()

    for step, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        #compute output
        logits = model(inputs)
        loss = criterion(logits, targets)

        ret = utils.accuracy(logits, targets, topk=(1, 5))
        n = inputs.size(0)
        total_loss.update(loss.item(), n)
        prec1 = ret[0]
        prec5 = ret[1]
        val_top1_accuracy.update(prec1.item(), n)
        val_top5_accuracy.update(prec5.item(), n)
        if step % 20 == 0:
            print("Valid %d %e %f %f" % (
                step, total_loss.average, val_top1_accuracy.average, val_top5_accuracy.average))
    return total_loss, val_top1_accuracy.average, val_top5_accuracy.average


def _main():
    args = get_argument()
    print(args)
    batch_size=args.batch_size
    model_path=args.model_path
    if os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    model=MobileNetV3_Large(num_classes=10)
    model.cuda()
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=3e-4)
    best_val_acc1=0

    train_loader,val_loader=dataSet.load_data(args)

    num_epoches=args.iter_epoch

    for epoch in range(num_epoches):
        start_time=time.time()
        train_loss,train_top1_accuracy,train_top5_accuracy=train(train_loader,model,optimizer,criterion)
        val_loss,val_top1_accuracy,val_top5_accuracy=valid(val_loader,model,criterion)
        is_best=val_top1_accuracy > best_val_acc1
        best_val_acc1=max(val_top1_accuracy,best_val_acc1)

        if is_best:
            print("saving...")
            best_acc5=val_top5_accuracy
            state={
                "model":model.state_dict(),
                "best_acc1":best_val_acc1,
                "best_acc5":best_acc5,
                "epoch":epoch
            }
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            filename = os.path.join(model_path, "checkpoint_epoch_{}.pth".format(epoch))
            torch.save(state, filename)
        end_time=time.time()
        print("num_time %d",(end_time-start_time))

def get_total_model_parameters(model):
    total_parameters=0
    for layer in list(model.parameters()):
        layer_parameter=1
        for l in layer.size():
            layer_parameter*=l
        total_parameters+=layer_parameter
    return total_parameters

if __name__ == '__main__':
   #model=MobileNetV3_Large()

   ##input=torch.randn(3,3,224,224)
  # se = SizeEstimator(model, input_size=input)
   #print(se.estimate_size())
    #output=model(input)
    #print(output.shape)
    #total_parameters=get_total_model_parameters(model)
    #print(total_parameters)

    #print(model)
    _main()

