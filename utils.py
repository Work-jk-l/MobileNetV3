import os
import torch

#utils.py：compute the average and current value
class AverageMeter():
    def __init__(self):
        self.average=0
        self.sum=0
        self.count=0
        self.value=0

    def update(self,value,n):
        self.value=value
        self.sum+=float(value*n)
        self.count+=n
        self.average=self.sum/self.count


def accuracy(logits,targets,topk=(1,)):
    #compute the correct topK
    maxK=max(topk) #取top1和top5的准确率
    value,pred=logits.topk(maxK,1,True,True) # torch.topK() dims=1按每行进行排序，返回每行的前k个最大值和对应的索引
    pred=pred.t().type_as(targets)
    correct=pred.eq(targets.view(1,-1).expand_as(pred)) # expand_as(),把一个tensor变成和函数括弧内一样形状的tensor

    res=[]
    for k in topk:
        correct_k=correct[:k].view(-1,1).float().sum(0,keepdim=True)
        res.append(correct_k.mul_(100.0/logits.size(0)))
    return res
