import torch
import numpy as np
import torch.nn as nn
import math
import pickle
import random
import torch.optim as optim
import argparse
import time
from sklearn.metrics import mean_squared_error


class DataLoader():
    def __init__(self, path):
        print("loading data ...")
        self.path = path
        self.statistics = dict()
        self.ratio = 0.2
        self.item_feature_dict = pickle.load(open(self.path + "item_feature_quality_matrix", "rb"))
        self.statistics = pickle.load(open(self.path + "statistics", "rb"))

        self.train_item_batch_all = []
        self.train_feature_batch_all = []
        self.train_score_batch_all = []

        self.test_item_batch_all = []
        self.test_feature_batch_all = []
        self.test_score_batch_all = []

    def generate_data(self):
        for k, v in self.item_feature_dict.items():
            for i in range(len(v)):
                if i <= int(len(v)*self.ratio) + 1:
                    self.train_item_batch_all.append(int(k))
                    self.train_feature_batch_all.append(int(v[i][0]))
                    self.train_score_batch_all.append(float(v[i][1]))
                else:
                    self.test_item_batch_all.append(int(k))
                    self.test_feature_batch_all.append(int(v[i][0]))
                    self.test_score_batch_all.append(float(v[i][1]))

        print('training sample number: ', len(self.train_item_batch_all))
        print('testing sample number: ', len(self.test_item_batch_all))



class UF_Att(nn.Module):
    def __init__(self, args, data):
        super(UF_Att, self).__init__()
        print('args: ', args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.data = data
        self.item_number = data.statistics['item_number']
        self.feature_number = data.statistics['feature_number']

        self.item_embedding_matrix = nn.Embedding(self.item_number, self.args.embedding_dim)
        self.feature_embedding_matrix = nn.Embedding(self.feature_number, self.args.embedding_dim)

        nn.init.uniform_(self.item_embedding_matrix.weight.data, a=0.0, b=1)
        nn.init.uniform_(self.feature_embedding_matrix.weight.data, a=0.0, b=1)
        # nn.init.normal_(self.user_embedding_matrix.weight.data, std=1. / math.sqrt(self.user_number))
        # nn.init.normal_(self.feature_embedding_matrix.weight.data, std=1. / math.sqrt(self.feature_number))

        self.criterion = nn.MSELoss()

    def forward(self, item_batch, feature_batch, score_batch):
        score_cuda = torch.FloatTensor(score_batch).to(self.device)
        eps = torch.FloatTensor([1e-6]).to(self.device)
        p_score = self.predict_score(item_batch, feature_batch)
        loss = torch.sqrt(self.criterion(p_score, score_cuda) + eps)
        return loss

    def predict_score(self, items, features):
        item_cuda = torch.LongTensor(items).to(self.device)
        feature_cuda = torch.LongTensor(features).to(self.device)

        item_embed = self.item_embedding_matrix(item_cuda)
        feature_embed = self.feature_embedding_matrix(feature_cuda)
        p_score = torch.mul(item_embed, feature_embed).mean(dim=1)

        return p_score


    def predict_all_matrix(self):
        all_matrix = torch.mul(torch.unsqueeze(self.item_embedding_matrix.weight.data, 1), torch.unsqueeze(self.feature_embedding_matrix.weight.data, 0)).mean(dim=2)
        return all_matrix




def parameter_parser():
    parser = argparse.ArgumentParser(description="Run SHCN.")

    parser.add_argument("--data_path",                 type=str,   default="../data/Automotive/",                help="data path")
    parser.add_argument("--embedding_dim",             type=int,   default=100,                            help="embedding dimension of the graph vertex")
    parser.add_argument("--epoch_number",              type=int,   default=1000,                            help="number of training epochs")
    parser.add_argument("--learning_rate",             type=float, default=1,                           help="learning rate")
    parser.add_argument("--batch_size",                type=int,   default=64,                            help="batch size")
    parser.add_argument("--optimizer",                 type=str,   default="sgd",                         help="adam, sgd, adadelta, adagrad or RMSprop")
    parser.add_argument("--reg",                       type=float, default=0.9,                       help="the regular item of the MF model")

    return parser.parse_args()



class experiment():
    def __init__(self, args, model, data):
        self.model = model
        self.data = data
        self.args = args

        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.reg)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)

    def shuffle(self, train_items, train_features, train_scores):
        train_records_num = len(train_items)
        index = np.array(range(train_records_num))
        np.random.shuffle(index)
        input_items = list(np.array(train_items)[index])
        input_features = list(np.array(train_features)[index])
        input_scores = list(np.array(train_scores)[index])
        return input_items, input_features, input_scores

    def run(self):
        result = []
        min_rmse = 1000
        best_result_index = 0
        train_items = self.data.train_item_batch_all
        train_features = self.data.train_feature_batch_all
        train_scores = self.data.train_score_batch_all

        for epoch in range(self.args.epoch_number):
            print('********************* training epoch begin *********************')
            step = 0
            train_record_number = len(self.data.train_item_batch_all)
            print('training sample number: ', train_record_number)

            train_items, train_features, train_scores = self.shuffle(train_items, train_features, train_scores)

            max_steps = train_record_number / self.args.batch_size
            self.model.train()

            s = time.time()
            while step <= max_steps:
                if (step + 1) * self.args.batch_size > train_record_number:
                    b = train_record_number - step * self.args.batch_size
                else:
                    b = self.args.batch_size
                start = step * self.args.batch_size
                end = start + b
                if end > start:
                    self.optimizer.zero_grad()
                    loss = self.model(train_items[start:end], train_features[start:end], train_scores[start:end])
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                step += 1
            print('epoch training time: \t', (time.time() - s), ' sec')
            print('********************* training epoch end *********************')

            print('&&&&&&&&&&&&&&&&&&&& test epoch begin &&&&&&&&&&&&&&&&&&&&')
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                rmse = self.performance_eval(self.model)

                end_time = time.time()
                result.append(rmse)
                if rmse < min_rmse:
                    min_rmse = rmse
                    best_result_index = epoch
                    self.best_item_feature_matrix = self.model.predict_all_matrix()

                print('current best performance: ' + str(result[best_result_index]))
                print('epoch:', epoch, 'testing performance: ', rmse, 'testing time: \t', (end_time - start_time), ' sec')
                print('&&&&&&&&&&&&&&&&&&&& test epoch end &&&&&&&&&&&&&&&&&&&&')

        print('-------------------- learning end --------------------')
        print(result)
        print(best_result_index)
        print('final best performance: ' + str(result[best_result_index]))
        return result[best_result_index], result


    def performance_eval(self, model):
        p_scores = model.predict_score(self.data.test_item_batch_all, self.data.test_feature_batch_all)
        rmse = mean_squared_error(self.data.test_score_batch_all, p_scores.cpu().data.numpy().tolist())
        return rmse


    def generate_final_user_feature_matrix(self):
        know_item_feature_dict = self.data.item_feature_dict
        predict_item_feature_matrix = self.best_item_feature_matrix.cpu().data.numpy()
        combined_matrix = dict()

        for item in range(len(predict_item_feature_matrix)):
            tmp = predict_item_feature_matrix[item]
            if str(item) in know_item_feature_dict.keys():
                for fs in know_item_feature_dict[str(item)]:
                    tmp[int(fs[0])] = fs[1]
                combined_matrix[item] = tmp

        pickle.dump(combined_matrix, open(self.args.data_path + 'predicted_item_feature_quality', 'wb'))

        # for k, v in know_item_feature_dict.items():
        #     print(k, v)
        #     print(combined_matrix[int(k)])
        #     input()



if __name__ == "__main__":
    '''
    The entrance of the application.
    (1) building dataset
    (2) defining model
    (3) training the model based on the dataset
    '''
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    args = parameter_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("current envirment: " + str(device))

    # (1) building dataset
    data = DataLoader(args.data_path)
    data.generate_data()
    # (2) defining model
    model = UF_Att(args, data).to(device)
    # (3) training the model based on the dataset.
    exp = experiment(args, model, data)
    r, r_epochs = exp.run()
    print(r, r_epochs)
    exp.generate_final_user_feature_matrix()





