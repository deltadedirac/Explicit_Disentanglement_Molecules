import torch
from torch import autograd
from torch import nn
import pdb
import torch.nn.functional as F
'''
    The inheritance to nn.Module is made with the purpose of obtaining 
    flexibility when it will be necessary to do a custom Loss Function
'''

class LossFunctionsAlternatives(nn.Module):
    """
    - This criterion choose a specifical loss function for optimizing the diffeomorphical transformation.
    
    - The inheritance of nn.Module template is just for creating customized loss functions in case 
      of being necessary.
    """

    def __init__(self):
        super().__init__()
        self.Cross_Entropy = nn.CrossEntropyLoss(reduction = 'none')
        self.L1Loss = nn.L1Loss()
        self.kl_div = nn.KLDivLoss()
        self.MSE = nn.MSELoss()
        self.softmax = nn.Softmax(dim=-1)
        self.logGauss = nn.GaussianNLLLoss()

    def get_dictionaries_to_mask_data(self, c2i, i2c, i2i):
        self.c2i = c2i
        self.i2c = i2c
        self.i2i = i2i

    def _JSD(self,input,target):
        pa, qa = input, target #input.view(-1, input.size(-1)), target.view(-1, target.size(-1))
        m = (0.5 * (pa + qa))
        loss  = 0.5 * (self.kl_div(m,pa) + self.kl_div(m,qa))
        return loss.mean()

    def forward(self, method, input, target, forw_per, **kargs):
        #pdb.set_trace()
        if 'masked_idx'  in kargs:
            masked_idx = kargs['masked_idx']

        if 'variational' in kargs:
            tuple_variational = kargs['variational']
        if 'target2' in kargs:
            target2 = kargs['target2']
        if 'input2' in kargs:
            input2 = kargs['input2']


        if method == 'CE':
            #pdb.set_trace()
            self.component_vals = self.Cross_Entropy(input.permute(forw_per), target.argmax(-1))
            self.loss= self.component_vals.sum(dim=-1).mean()
        elif method == 'CEmask':
            #print('from here')
            self.loss= self.Cross_Entropy(input[: , masked_idx].permute(forw_per), target[: , masked_idx].argmax(-1))
            #print('done')

        elif method == 'L1':
            self.loss = self.L1Loss(input,target)
        elif method == 'KL':
            self.loss = self.kl_div(input,target)
        elif method == 'MSE':
            self.loss = self.MSE(input,target)
        elif method == 'logGauss':
            self.loss = self.logGauss(input,target, torch.ones(*input.shape, requires_grad=True))
        elif method == 'Soft_Label_KLD':
            log_probs = F.log_softmax(input,dim=-1)
            self.component_vals = self.kl_div(log_probs,target)
            self.loss = self.component_vals.mean()
        elif method == 'JSD':
            #import pdb; pdb.set_trace()
            pa, qa = input, target #input.view(-1, input.size(-1)), target.view(-1, target.size(-1))
            m = (0.5 * (pa + qa))
            self.loss  = 0.5 * (self.kl_div(m,pa) + self.kl_div(m,qa))
            self.loss = self.loss.mean()
        elif method == 'CustomVariational_1':
            self.component_vals1 = self.Cross_Entropy(input.permute(forw_per), target.argmax(-1))
            self.component_vals2 = self.kl_div(tuple_variational[5],target2) #self.Cross_Entropy(tuple_variational[5].permute(forw_per), target2.argmax(-1))
            self.loss = self.component_vals1.mean() + self.component_vals2.mean()
        elif method == 'Guided_Soft_Label_KLD':
            log_probsinput = F.log_softmax(input,dim=-1)
            self.component_vals1 = self.kl_div(log_probsinput,target)
            log_probscomp = F.log_softmax(tuple_variational[5],dim=-1)
            self.component_vals2 = self.kl_div(log_probscomp,target2)
            self.loss = self.component_vals1.mean() + self.component_vals2.mean()
        elif method == 'Guided_JSD':
            JSD1 = self._JSD(input,target)
            JSD2 = self._JSD(tuple_variational[5],target2)
            self.loss = JSD1 + JSD2
        elif method == 'CustomVariational_2': 
            #SoftLabel = self.Cross_Entropy(input.permute(forw_per), target.argmax(-1))
            SoftLabel = self.kl_div(  F.log_softmax(input,dim=-1)  ,target).mean()
            diff_thetas = self.kl_div( input2 , target2 ).mean()
            self.loss = SoftLabel + diff_thetas
        elif method =='FocalLoss':
            log_prob = F.log_softmax(input, dim=-1)
            prob = torch.exp(log_prob)
            loss = F.nll_loss(
                ((1 - prob) ** self.gamma) * log_prob, 
                target, 
                weight=self.weight,
                reduction = self.reduction
            )

        else:
            self.loss = None


            
        return self.loss
