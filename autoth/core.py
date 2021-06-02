import numpy as np
import time
import torch.nn as nn
import torch
from sklearn import metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HyperParamsOptimizer(object):
    def __init__(self, score_calculator, learning_rate=1e-2, epochs=100, 
        step=0.01, max_search=5):
        """Hyper parameters optimizer. Parameters are optimized using gradient
        descend methods by using the numerically calculated graident: 
        gradient: f(x + h) - f(x) / (h)

        Args:
          score_calculator: object. See ScoreCalculatorExample in example.py as 
              an example.
          learning_rate: float
          epochs: int
          step: float, equals h for calculating gradients
          max_search: int, if plateaued, then search for at most max_search times
        """
        
        self.score_calculator = score_calculator
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = Adam()
        self.optimizer.alpha = learning_rate
        self.step = step
        self.max_search = max_search

    def do_optimize(self, init_params, pred, target):
        params = init_params.clone().detach()

        t1 = time.time()
        for i in range(self.epochs):
            (score, grads) = self.calculate_gradients(params, pred, target)
            grads = (-grads)
            params = self.optimizer.GetNewParams(params, grads)
            
        #print('    Hyper params score: {:.4f}'.format(score))
        #print('    Epoch: {}, Time: {:.4f} s'.format(i, time.time() - t1))
        
        return score, params

    def calculate_gradients(self, params, pred, target):
        score = self.score_calculator(params, pred, target)
        step = self.step

        grads=None
        
        new_params = params.clone()
        cnt = 0
        
        params = params.squeeze()
        for k, param in enumerate(params):
            new_params = params.clone()
            cnt = 0
            while cnt < self.max_search:
                cnt += 1
                new_params[k] += self.step
                new_score = self.score_calculator(new_params , pred, target)

                if new_score != score:
                    break
            grad = (new_score - score) / (step * cnt)
            grad = grad.unsqueeze(0)
            if grads is None:
                grads = grad
            else:
                grads = torch.cat((grads,grad),0)

        grads = grads.squeeze()
        return score, grads


class Base(nn.Module):
    def _reset_memory(self, memory):
        for i1 in range(len(memory)):
            memory[i1] = torch.zeros(memory[i1].shape)


class Adam(Base):
    def __init__(self):
        self.ms = None
        self.alpha = 1e-3
        self.beta1 = torch.Tensor([[0.9]]).to(device)
        self.beta2 = torch.Tensor([[0.999]]).to(device)
        self.eps = 1e-8
        self.iter = 0
        
    def GetNewParams(self, params, gparams):
        if self.ms is None:
            self.ms = torch.zeros_like(params)
            self.vs = torch.zeros_like(params)
          
        # fast adam, faster than origin adam
        self.iter += 1
        alpha_t = self.alpha * torch.sqrt(1 - torch.pow(self.beta2, self.iter)) / (1 - torch.pow(self.beta1, self.iter))
        
        self.ms = torch.reshape(self.ms,(1,gparams.shape[-1]))
        self.vs = torch.reshape(self.vs,(1,gparams.shape[-1]))
        
        temp1 =torch.mm(self.beta1, self.ms)
        temp2 = torch.mm((1-self.beta1),gparams.unsqueeze(0).to(torch.float))
        self.ms = torch.add(temp1, temp2)
        
        self.vs = self.beta2 * self.vs + (1 - self.beta2) * torch.square(gparams)

        new_params = params - alpha_t * self.ms / (torch.sqrt(self.vs + self.eps))
        new_params = new_params.squeeze()
        
        return new_params
        
    def reset(self):
        self._reset_memory(self.ms)
        self._reset_memory(self.vs)
        self.epoch = 1
        
class ScoreCalculatorExample(object):
    def __init__(self, batch_size, classes_num):
        """An example of Score calculator. Used to calculate score (such as F1), 
        given prediction, target and hyper parameters. 
        """
        self.N = batch_size    # Number of samples
        self.classes_num = classes_num    # Number of classes

    def __call__(self, params, prediction, target):
        """Parameters (such as thresholds) are used calculate score.
        Args:
          params: list of float
        Returns:
          score: float
        """
        thresholds = params
        output = torch.zeros_like(prediction)
        
        if self.N > prediction.size(0):
            batch_size = prediction.size(0)
        else:
            batch_size = self.N
            
        # Threshold to output
        output = torch.where(prediction > torch.tile(thresholds,(batch_size,1)), 1, 0)
        
        # Calculate score
        tp = (output * target).sum(1)
        fp = (output * (1 - target)).sum(1)
        fn = ((1 - output) * target).sum(1)
        f1 = tp / (tp + (fp + fn) / 2)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        #return f1.mean(), precision.mean(), recall.mean()

        return f1.mean()