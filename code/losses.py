import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies
        
        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings = False)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss

def distMC(Mat_A, Mat_B, norm=1, cpu=False, sq=True):#N by F
    N_A = Mat_A.size(0)
    N_B = Mat_B.size(0)
    
    DC = Mat_A.mm(torch.t(Mat_B))
    if cpu:
        if sq:
            DC[torch.eye(N_A).bool()] = -norm
    else:
        if sq:
            DC[torch.eye(N_A).bool().cuda()] = -norm
            
    return DC

def Mat(Lvec):
    N = Lvec.size(0)
    Mask = Lvec.repeat(N,1)
    Same = (Mask==Mask.t())
    return Same.clone().fill_diagonal_(0), ~Same#same diff
    
class HSELoss(nn.Module):
    def __init__(self,s=0.1):
        super(HSELoss, self).__init__()
        self.semi = False
        self.sigma1 = s
        self.sigma2 = s
        
    def forward(self, fvec, Lvec,IPC,bz,a,data,loss):
        N = Lvec.size(0)
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)
        # matting
        Same, Diff = Mat(Lvec.view(-1))
        # Similarity Matrix
        Dist = distMC(fvec_norm,fvec_norm)
        D_detach_P = Dist.clone().detach()
        D_detach_P[Diff]=-1
        D_detach_P[D_detach_P>0.9999]=-1
  
        D_detach_con_easy = Dist.clone().detach()
        D_detach_con_Hard = Dist.clone().detach()

        if(data!='SOP'):
            # Find weak positive for HS
            for i in range(bz-IPC,bz):
                for j in range(2*IPC,bz):
                    D_detach_con_easy[i][j]=-1
            V_con_easy,I_con_easy=D_detach_con_easy.max(1)

            # Find negative for HS
            for i in range(bz-IPC,bz):
                for j in range(2*IPC):
                    D_detach_con_Hard[i][j]=-1
                for j in range(bz-IPC,bz):
                    D_detach_con_Hard[i][j]=-1
            V_con_Hard,I_con_Hard=D_detach_con_Hard.max(1)
        else:
            for i in range(120,180):
                for j in range(120,180):
                    D_detach_con_easy[i][j]=-1
                for y in range(120):
                    if(Lvec[(i-120)//5*10+i%5]!=Lvec[y] and Lvec[(i-120)//5*10+i%5+5]!=Lvec[y]):
                        D_detach_con_easy[i][y]=-1
            V_con_easy,I_con_easy=D_detach_con_easy.max(1)

            for i in range(120,180):
                for j in range(120,180):
                    D_detach_con_Hard[i][j]=-1      
            for x in range(120,180):
                for y in range(120):
                    if(Lvec[(x-120)//5*10+x%5]==Lvec[y] or Lvec[(x-120)//5*10+x%5+5]==Lvec[y]):
                        D_detach_con_Hard[x][y]=-1
            if(loss=='Proxy_Anchor'):
                for n in range(12):
                    for x in range(120+n*5,120+(n+1)*5):
                        for y in range(n*10):
                            D_detach_con_Hard[x][y]=-1
                        for y in range(n*10+10,120):
                            D_detach_con_Hard[x][y]=-1
            V_con_Hard,I_con_Hard=D_detach_con_Hard.max(1)         
            
        VCE = Dist[torch.arange(0,N), I_con_easy]
        VCH = Dist[torch.arange(0,N), I_con_Hard]
        T1=torch.stack([VCE,VCH],1)
        if(data!='SOP'):
            T1=T1[-IPC:]
        else:
            T1=T1[-60:]
        Prob1 = -F.log_softmax(T1/self.sigma1,dim=1)[:,0]
        loss=a*Prob1.mean()
        return loss

class EPHNLoss(nn.Module):
    def __init__(self,s=0.1):
        super(EPHNLoss, self).__init__()
        self.semi = True
        self.sigma = s
        
    def forward(self, fvec, Lvec):
        N = Lvec.size(0)
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)
        # matting
        Same, Diff = Mat(Lvec.view(-1))
        
        # Similarity Matrix
        Dist = distMC(fvec_norm,fvec_norm)
        
        ############################################
        # finding max similarity on same label pairs
        D_detach_P = Dist.clone().detach()
        D_detach_P[Diff]=-1
        D_detach_P[D_detach_P>0.9999]=-1
        V_pos, I_pos = D_detach_P.max(1)
 
        # prevent duplicated pairs
        Mask_not_drop_pos = (V_pos>0)

        # extracting pos score
        Pos = Dist[torch.arange(0,N), I_pos]
        Pos_log = Pos.clone().detach().cpu()
        
        ############################################
        # finding max similarity on diff label pairs
        D_detach_N = Dist.clone().detach()
        D_detach_N[Same]=-1
        if self.semi:
            D_detach_N[(D_detach_N>(V_pos.repeat(N,1).t()))&Diff]=-1#extracting SHN
        V_neg, I_neg = D_detach_N.max(1)
            
        # prevent duplicated pairs
        Mask_not_drop_neg = (V_neg>0)

        # extracting neg score
        Neg = Dist[torch.arange(0,N), I_neg]
        Neg_log = Neg.clone().detach().cpu()
        
        # triplets
        T = torch.stack([Pos,Neg],1)
        Mask_not_drop = Mask_not_drop_pos&Mask_not_drop_neg

        # loss
        Prob = -F.log_softmax(T/self.sigma,dim=1)[:,0]
        loss = Prob[Mask_not_drop].mean()

        return loss

class EDMLoss(nn.Module):
    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self,p_target:torch.Tensor,p_estimate:torch.Tensor):
        assert p_target.shape==p_estimate.shape
        cdf_target=torch.cumsum(p_target,dim=1)
        cdf_estimate=torch.cumsum(p_estimate,dim=1)
        cdf_diff=cdf_estimate-cdf_target

        samplewisd_emd=torch.sqrt( torch.mean(torch.pow(torch.abs(cdf_diff),2)))

        return samplewisd_emd.mean()

