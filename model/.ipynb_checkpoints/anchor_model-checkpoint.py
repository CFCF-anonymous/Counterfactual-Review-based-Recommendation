import torch
import torch.nn as nn
import math


class Anchor(nn.Module):

    def __init__(self, args, data):
        super(Anchor, self).__init__()
        print('args: ', args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self._temp = 0.1
        self.data = data
        self.user_number = data.statistics['user_number']
        self.item_number = data.statistics['item_number']
        self.feature_number = data.statistics['feature_number']

        self.user_embedding_matrix = nn.Embedding(self.user_number, self.args.embedding_dim)
        self.item_embedding_matrix = nn.Embedding(self.item_number, self.args.embedding_dim)
        self.feature_embedding_matrix = nn.Embedding(self.feature_number, self.args.f_embedding_dim)

        torch.nn.init.uniform_(self.user_embedding_matrix.weight.data, a=0, b=0.1)
        torch.nn.init.uniform_(self.item_embedding_matrix.weight.data, a=0, b=0.1)
        torch.nn.init.uniform_(self.feature_embedding_matrix.weight.data, a=0, b=0.1)
        
        # torch.nn.init.normal_(self.user_embedding_matrix.weight.data, std=1. / math.sqrt(self.user_number))
        # torch.nn.init.normal_(self.item_embedding_matrix.weight.data, std=1. / math.sqrt(self.user_number))
        # torch.nn.init.normal_(self.feature_embedding_matrix.weight.data, std=1. / math.sqrt(self.user_number))

        self.user_map = nn.Parameter(torch.empty(self.feature_number, self.args.f_embedding_dim, device=self.device), requires_grad=True)
        torch.nn.init.uniform_(self.user_map, a=-0.001, b=-0.001)
        self.item_map = nn.Parameter(torch.empty(self.feature_number, self.args.f_embedding_dim, device=self.device), requires_grad=True)
        torch.nn.init.uniform_(self.item_map, a=-0.001, b=-0.001)

        self._cnt = 0

        self.bi_cross_entropy = torch.nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.args.f_embedding_dim + self.args.embedding_dim, 10)
        nn.init.normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

        self.bn = torch.nn.BatchNorm1d(self.args.f_embedding_dim + self.args.embedding_dim)

        self.fc2 = nn.Linear(10, 1)
        nn.init.normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)


    def forward(self, user_batch, user_feature_batch, pos_item_batch, pos_item_feature_batch, neg_item_batch, neg_item_feature_batch):
        """ 
            Remove the softmax
        """
        # convert nunmpy to tensors
        user_id_t = torch.LongTensor(user_batch).to(self.device)
        user_feature_t = torch.FloatTensor(user_feature_batch).to(self.device)
        pos_item_id_t = torch.LongTensor(pos_item_batch).to(self.device)
        pos_item_feature_t = torch.FloatTensor(pos_item_feature_batch).to(self.device)
        neg_item_id_t = torch.LongTensor(neg_item_batch).to(self.device)
        neg_item_feature_t = torch.FloatTensor(neg_item_feature_batch).to(self.device)

        # compute positive socre
        pos_score = self.predict_score(user_id_t, user_feature_t, pos_item_id_t, pos_item_feature_t)
        # compute negative socre
        neg_score = self.predict_score(user_id_t, user_feature_t, neg_item_id_t, neg_item_feature_t)
        # compute loss
        loss = - torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
        #print ("Logits: ", (pos_score - neg_score)[0])
        return loss


    def predict_score_xk(self, users, user_features, items, item_features):
        user_embed = self.user_embedding_matrix(users)
        item_embed = self.item_embedding_matrix(items)
        u_i = torch.mul(user_embed, item_embed)

        # user_feature_norm = torch.softmax(torch.mul(user_features, item_features), dim=1)
        user_feature_norm = torch.softmax(user_features, dim=1)
        user_mapped = torch.matmul(user_feature_norm, self.user_map)
        item_feature_norm = torch.softmax(item_features, dim=1)
        item_mapped = torch.matmul(item_feature_norm, self.item_map)
        ui_feature = torch.mul(user_mapped, item_mapped)

        #ui_feature_cat = torch.cat((u_i, ui_feature), dim=1)
        result = self.fc2(self.relu(self.fc1(ui_feature)))
        return result

    def feature_normalize(self, feature_vector):
        """ 
            normalization, [0 - 5] -> [-1, 1]
        """
        tmp = (feature_vector - 2.5) / 2.5
        return tmp

    def predict_score(self, users, user_features, items, item_features):
        #import pdb
        #pdb.set_trace()
        user_embed = self.user_embedding_matrix(users)
        item_embed = self.item_embedding_matrix(items)
        u_i = torch.mul(user_embed, item_embed)

        """
            origin : 
        """
        #user_feature_norm = torch.softmax(user_features / self._temp, dim=1)
        #item_feature_norm = torch.softmax(item_features / self._temp, dim=1)
        """
            remove softmax : 
        """
        
        user_feature_norm = self.feature_normalize(user_features)
        item_feature_norm = self.feature_normalize(item_features)
        user_mapped = torch.matmul(user_feature_norm, self.user_map)
        item_mapped = torch.matmul(item_feature_norm, self.item_map)
        
        ui_feature = torch.mul(user_mapped, item_mapped)
        ui_feature_cat = torch.cat((u_i, ui_feature), dim=1)
        #ui_feature_cat = self.bn(ui_feature_cat)
        result = self.fc2(self.relu(self.fc1(ui_feature_cat)))
        #self._cnt += 1
        #if (self._cnt % 50 == 0):
            #print ('feature_cat ', ui_feature_cat)
            #print (result.max())
        #    pass
        
        return result





    def predict_score_2(self, users, user_features, items, item_features):

        user_embed = self.user_embedding_matrix(users)
        item_embed = self.item_embedding_matrix(items)
        u_i = torch.mul(user_embed, item_embed)

        feature_similarity = torch.softmax(torch.mul(user_features, item_features), dim=1)
        weighted_feature_embed = torch.mul(torch.unsqueeze(feature_similarity, dim=2), self.feature_embedding_matrix.weight)
        weighted_feature_embed = weighted_feature_embed.sum(dim=1)
        ui_feature_cat = torch.cat((u_i, weighted_feature_embed), dim=1)

        ui_feature_cat = bn(ui_feature_cat)
        result = self.fc2(self.relu(self.fc1(ui_feature_cat)))
        return result

    def predict_score_1(self, users, user_features, items, item_features):

        user_embed = self.user_embedding_matrix(users)
        item_embed = self.item_embedding_matrix(items)
        feature_similarity = torch.softmax(torch.mul(user_features, item_features), dim=1)
        weighted_feature_embed = torch.mul(torch.unsqueeze(feature_similarity, dim=2), self.feature_embedding_matrix.weight)

        ui_interaction = torch.mul(user_embed, item_embed).sum(dim=1)
        user_feature_interaction = torch.mul(torch.unsqueeze(user_embed, 1), weighted_feature_embed).sum(dim=2).sum(dim=1)
        item_feature_interaction = torch.mul(torch.unsqueeze(item_embed, 1), weighted_feature_embed).sum(dim=2).sum(dim=1)
        feature_interaction = 0.5*torch.mul(weighted_feature_embed.sum(dim=1), weighted_feature_embed.sum(dim=1)).sum(dim=1)\
                              - torch.mul(weighted_feature_embed, weighted_feature_embed).sum(dim=2).sum(dim=1)

        result = ui_interaction + user_feature_interaction + item_feature_interaction + feature_interaction

        return result


