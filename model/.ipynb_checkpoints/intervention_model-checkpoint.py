import torch
import torch.nn as nn
import math

torch.manual_seed(2021)


class Intervener(nn.Module):

    def __init__(self, args, data):
        super(Intervener, self).__init__()
        print('args: ', args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.data = data
        self.user_number = data.statistics['user_number']
        self.item_number = data.statistics['item_number']
        self.feature_number = data.statistics['feature_number']

        self.tau = nn.Parameter(torch.empty(self.args.intervener_batch_size, self.feature_number, device=self.device), requires_grad=True)
        torch.nn.init.uniform_(self.tau, a=-1, b=1)

        # self.tau_embedding_matrix = nn.Embedding(self.user_number, self.feature_number)
        # torch.nn.init.uniform_(self.tau_embedding_matrix.weight.data, a=0, b=0.1)
        # self.f_m = nn.Parameter(torch.empty(self.feature_number, self.feature_number, device=self.device), requires_grad=True)
        # torch.nn.init.uniform_(self.f_m, a=0, b=0.1)

        self.bi_cross_entropy = torch.nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def reset(self):
        torch.nn.init.uniform_(self.tau, a=-1, b=1)
        
    def set_anchor(self, anchor):
        self.anchor = anchor
        self.anchor.eval()
        # requires_grad set to False
        for name, param in self.anchor.named_parameters():
            param.requires_grad = False

    def forward(self, user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch, neg_item_batch, neg_item_feature_batch, fid=-1):
        user_id_t = torch.LongTensor(user_batch).to(self.device)
        user_feature_t = torch.FloatTensor(user_feature_batch).to(self.device)
        # generate mask
        import pdb
        #pdb.set_trace()
        if not self.args.intervener_soft: 
            if (fid == -1) :
                index = torch.topk(user_feature_t, self.args.intervener_feature_number, dim=1)[1]
                self.mask = torch.zeros(self.args.intervener_batch_size, self.feature_number).to(self.device)
                self.mask.scatter_(1, index, 1.)
                self.masked_tau = torch.mul(self.tau, self.mask)
            elif (fid >= 0):
                self.mask = torch.zeros(self.args.intervener_batch_size, self.feature_number).to(self.device)
                self.mask.scatter_(1, torch.ones(self.args.intervener_batch_size, 1).long().cuda() * fid, 1.)
                self.masked_tau = torch.mul(self.tau, self.mask)
        else : 
            self.masked_tau = self.tau
        # generate new feature
        user_feature_t = torch.add(user_feature_t, self.masked_tau)

        pos_item_id_t = torch.LongTensor(pos_item_batch).to(self.device)
        pos_item_feature_t = torch.FloatTensor(pos_item_feature_batch).to(self.device)
        neg_item_id_t = torch.LongTensor(neg_item_batch).to(self.device)
        neg_item_feature_t = torch.FloatTensor(neg_item_feature_batch).to(self.device)

        #user_feature_t.retain_grad()
        pos_score = self.anchor.predict_score(user_id_t, user_feature_t, pos_item_id_t, pos_item_feature_t)
        neg_score = self.anchor.predict_score(user_id_t, user_feature_t, neg_item_id_t, neg_item_feature_t)
        #print ("Score:", pos_score - neg_score)
        conf =     - torch.nn.functional.logsigmoid(neg_score - pos_score)
        loss_sum = conf.sum()
        
        loss_sum += self.args.intervener_reg * torch.norm(self.masked_tau, 2)
        if self.args.intervener_soft: 
            loss_sum += self.args.intervener_l1_reg * torch.norm(self.masked_tau, 1)
        
        #import pdb
        #pdb.set_trace()
        #loss_sum = torch.norm(self.masked_tau, 1)
        #print ("number conf ", (conf<0.693).sum().item())
        #print ("Max    conf ", (conf).max().item())
        #print ("Mean   conf ", (conf).mean().item())
        #print ("zero rate: ", 1.0 * torch.sum(torch.abs(self.masked_tau)<1e-2) / len(self.masked_tau.flatten()))
        return loss_sum, conf # conf > 0 and conf < 0.693 is nice
