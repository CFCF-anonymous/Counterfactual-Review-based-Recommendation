import torch
import numpy as np
import torch.optim as optim
import time
from model.CounterFea import CounterFRec
Anchor = None
from model.intervention_model import Intervener
from data_loader import DataLoader
import argparse
import random
from utils.eval import Evaluate


# python3 main.py --anchor_model 1  --data_path ./data/Amazon_Instant_Video/
# --f_embedding_dim 90 --batch_size 32 --reg 0.0001 --learning_rate 0.01 --embedding_dim 50



def parameter_parser():
    parser = argparse.ArgumentParser(description="Run SHCN.")

    parser.add_argument("--data_path",                 type=str,   default="./data/Amazon_Instant_Video/",  help="data path")
    parser.add_argument("--embedding_dim",             type=int,   default=50,                              help="embedding dimension of the graph vertex")
    parser.add_argument("--f_embedding_dim",           type=int,   default=90,                              help="embedding dimension of the graph vertex")
    parser.add_argument("--K",                         type=int,   default=10,                              help="recommending how many items to a user")
    parser.add_argument("--epoch_number",              type=int,   default=40,                              help="number of training epochs")
    parser.add_argument("--learning_rate",             type=float, default=0.02,                          help="learning rate")
    parser.add_argument("--batch_size",                type=int,   default=32,                              help="batch size")
    parser.add_argument("--optimizer",                 type=str,   default="sgd",                           help="adam, sgd, adadelta, adagrad or RMSprop")
    parser.add_argument("--reg",                       type=float, default=0.0025,                          help="the regular item of the MF model")

    parser.add_argument("--confidence",                type=float, default=0.55,                            help="should small than -0.6931471805599453")
    parser.add_argument("--intervener_feature_number", type=int,   default=60,                              help="recommending how many items to a user")
    parser.add_argument("--intervener_iteration",      type=int,   default=1000,                            help="number of training epochs")
    parser.add_argument("--intervener_learning_rate",  type=float, default=0.001,                           help="learning rate")
    parser.add_argument("--intervener_batch_size",     type=int,   default=50,                              help="tau batch size")
    parser.add_argument("--intervener_reg",            type=float, default=0.01,                            help="the regular item of the MF model")
    parser.add_argument("--intervener_l1_reg",         type=float, default=0.0025,                          help="the regular item of the MF model")
    parser.add_argument("--intervener_soft",           type=bool,  default=False,                           help="the regular item of the MF model")
    parser.add_argument("--anchor_model",              type=int,   default=1,                               help="the regular item of the MF model")
    #parser.add_argument("--anchor",                    type=str,   default='ele_add',                       help="the type of the anchor model")
    parser.add_argument("--case_model",                type=int,   default=-1,                              help="the regular item of the MF model")
    parser.add_argument("--balanced_multiply",         type=float, default=1.0,                             help="the regular item of the MF model")

    return parser.parse_args()



class experiment():

    def __init__(self, args, a_model, i_model, data):
        self.anchor = a_model
        self.intervener = i_model
        self.data = data
        self.args = args
        self.evaluator = Evaluate(self.args.K)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.args.optimizer == 'adam':
            self.anchor_optimizer = torch.optim.Adam(self.anchor.parameters(), lr=self.args.learning_rate, weight_decay=self.args.reg)
            self.intervener_optimizer = torch.optim.Adam([self.intervener.tau], lr=self.args.intervener_learning_rate, weight_decay=0.0)
        elif self.args.optimizer == 'sgd':
            self.anchor_optimizer = torch.optim.SGD(self.anchor.parameters(), lr=self.args.learning_rate, weight_decay=self.args.reg, momentum=0.95)
            self.intervener_optimizer = torch.optim.SGD([self.intervener.tau], lr=self.args.intervener_learning_rate, weight_decay=0.0, momentum=0.95)

    def balanced_dataset(self, multiply=1.0):
        train_users = self.data.user_all
        train_pos_items = self.data.pos_item_all
        train_neg_items = self.data.neg_item_all
        user2idx = {}
        for idx, user in enumerate(train_users):
            user2idx[user] = user2idx.get(user, []) + [idx]
            
        avg_number = int((len(train_users) // len(user2idx)) * 1.0 * self.args.balanced_multiply)
        index = []
        import random
        for user, indices in user2idx.items():
            index.extend([random.choice(indices) for _ in range(avg_number)])
            
        print ("Balanced Dataset Len : ", len(index))
        import pdb
        return index
        
    
    def counter_data_generation(self, model):
        train_users = self.data.user_all
        train_users_feature = self.data.user_feature_all
        train_pos_items = self.data.pos_item_all
        train_pos_items_feature = self.data.pos_feature_all
        train_neg_items = self.data.neg_item_all
        train_neg_items_feature = self.data.neg_feature_all
        
        if model == 4:
            index = self.balanced_dataset(1.0)
            train_users = [train_users[idx] for idx in index]
            train_users_feature = [train_users_feature[idx] for idx in index]
            train_pos_items = [train_pos_items[idx] for idx in index]
            train_pos_items_feature = [train_pos_items_feature[idx] for idx in index]
            train_neg_items = [train_neg_items[idx] for idx in index]
            train_neg_items_feature = [train_neg_items_feature[idx] for idx in index]
            
            #self.data.user_all = train_users
            #self.data.user_feature_all = train_users_feature
            #self.data.pos_item_all = train_pos_items
            #self.data.pos_feature_all = train_pos_items_feature
            #self.data.neg_item_all = train_neg_items
            #self.data.neg_feature_all = train_neg_items_feature
        
        gen_train_users = []
        gen_train_users_feature = []
        gen_train_pos_items = []
        gen_train_pos_items_feature = []
        gen_train_neg_items = []
        gen_train_neg_items_feature = []

        step = 0
        train_record_number = len(train_users)
        max_steps = train_record_number / self.args.intervener_batch_size
        self.intervener.train()
        gen_number = 0
        while step <= max_steps:
            print('Data generation: ', step, max_steps)
            start = step * self.args.intervener_batch_size
            end = (step + 1) * self.args.intervener_batch_size
            if end < train_record_number:
                tau, conf = self.counter_data_generation_batch(train_users[start:end], train_users_feature[start:end],
                                  train_pos_items[start:end], train_pos_items_feature[start:end],
                                  train_neg_items[start:end], train_neg_items_feature[start:end])
                tau = tau.detach().cpu().numpy()
                conf = conf.detach().cpu().numpy()

                for i in range(start, end):
                    cf = conf[i-start]
                    if cf < self.args.confidence:
                            
                        gen_number += 1
                        if model == 3: # random
                            u = train_users[i]
                            u_f = list(np.array(train_users_feature[i]) + np.random.uniform(0, 5, tau[i-start].shape))
                            n = train_pos_items[i]
                            n_f = train_pos_items_feature[i]
                            p = train_neg_items[i]
                            p_f = train_neg_items_feature[i]
                        else:
                            u = train_users[i]
                            u_f = list(np.array(train_users_feature[i]) + tau[i-start])
                            n = train_pos_items[i]
                            n_f = train_pos_items_feature[i]
                            p = train_neg_items[i]
                            p_f = train_neg_items_feature[i]
                            

                        gen_train_users.append(u)
                        gen_train_users_feature.append(u_f)
                        gen_train_pos_items.append(p)
                        gen_train_pos_items_feature.append(p_f)
                        gen_train_neg_items.append(n)
                        gen_train_neg_items_feature.append(n_f)
            step += 1

        print('final generated sample number: ', gen_number)

        return gen_train_users, gen_train_users_feature, gen_train_pos_items, \
                gen_train_pos_items_feature, gen_train_neg_items, gen_train_neg_items_feature




    def counter_data_generation_batch(self, user, user_feature, pos_item, pos_item_feature, neg_item, neg_item_feature):
        min_loss = 100000
        for i in range(self.args.intervener_iteration):
            self.intervener_optimizer.zero_grad()
            loss, _ = self.intervener(user, user_feature, pos_item, pos_item_feature, neg_item, neg_item_feature, self.args.case_model)
            loss.backward(retain_graph=True)
            import pdb
            
            self.intervener_optimizer.step()

        loss, conf = self.intervener(user, user_feature, pos_item, pos_item_feature, neg_item, neg_item_feature, self.args.case_model)
        print ("loss: ", loss)
        if loss.detach().cpu().numpy() < min_loss:
            min_loss = loss.detach().cpu().numpy()
            final_tau = self.intervener.masked_tau
            final_conf = conf
            
        print ("Max Tau", final_tau.max())
        print ("Min Tau", final_tau.min())
        print ("Mean Conf", final_conf.mean())
        
        dataset = []
        if (self.args.case_model != -1):
            final_tau = final_tau.detach().cpu().numpy()[:,self.args.case_model]
            final_conf = final_conf.detach().cpu().numpy()[:,0]
            for idx, (i_tau, i_conf) in enumerate(zip(final_tau, final_conf)):
                if i_conf < 0.69:
                    dataset.append([user[idx], pos_item[idx], neg_item[idx], int(self.args.case_model), i_tau])
            import pickle as pkl
            pkl.dump(dataset, open('./result/Tools/{}.pkl'.format(self.args.case_model), 'wb'))
            exit(0)
                
        return final_tau, final_conf


    def load(self):
        print('model loading ...')
        self.anchor = torch.load(self.args.data_path + 'anchor_best.ptr')
        if self.args.optimizer == 'adam':
            self.anchor_optimizer = torch.optim.Adam(self.anchor.parameters(), lr=self.args.learning_rate, weight_decay=self.args.reg)
        elif self.args.optimizer == 'sgd':
            import pdb
            self.anchor_optimizer = torch.optim.SGD(self.anchor.parameters(), lr=self.args.learning_rate, weight_decay=self.args.reg, momentum=0.9)
        self.intervener.set_anchor(self.anchor)
        print (self.anchor)

    def save(self):
        print('model saving ...')
        torch.save(self.anchor, self.args.data_path + 'anchor.ptr')


    def shuffle(self, train_users, train_users_feature,
                      train_pos_items, train_pos_items_feature,
                      train_neg_items, train_neg_items_feature):
        train_records_num = len(train_users)
        index = np.array(range(train_records_num))
        np.random.shuffle(index)
        input_user = list(np.array(train_users)[index])
        input_user_feature = list(np.array(train_users_feature)[index])
        input_pos_item = list(np.array(train_pos_items)[index])
        input_pos_item_feature = list(np.array(train_pos_items_feature)[index])
        input_neg_item = list(np.array(train_neg_items)[index])
        input_neg_item_feature = list(np.array(train_neg_items_feature)[index])

        return input_user, input_user_feature, input_pos_item, input_pos_item_feature, input_neg_item, input_neg_item_feature

    def run(self, mode):
        if mode == 1:
            result = []
            max_value = 0
            best_result_index = 0
            train_users = self.data.user_all
            train_users_feature = self.data.user_feature_all
            train_pos_items = self.data.pos_item_all
            train_pos_items_feature = self.data.pos_feature_all
            train_neg_items = self.data.neg_item_all
            train_neg_items_feature = self.data.neg_feature_all

            for epoch in range(self.args.epoch_number):
                print('********************* training epoch begin *********************')
                step = 0
                train_record_number = len(train_users)
                print('training sample number: ', train_record_number)

                train_users, train_users_feature, train_pos_items, train_pos_items_feature, train_neg_items, train_neg_items_feature = \
                    self.shuffle(train_users, train_users_feature,
                                 train_pos_items, train_pos_items_feature,
                                 train_neg_items, train_neg_items_feature)

                max_steps = train_record_number / self.args.batch_size
                s = time.time()
                losses = []
                while step <= max_steps:
                    if (step + 1) * self.args.batch_size > train_record_number:
                        b = train_record_number - step * self.args.batch_size
                    else:
                        b = self.args.batch_size
                    start = step * self.args.batch_size
                    end = start + b
                    if end > start:
                        self.anchor.train()
                        self.anchor_optimizer.zero_grad()
                        loss = self.anchor(train_users[start:end], train_users_feature[start:end],
                                          train_pos_items[start:end], train_pos_items_feature[start:end],
                                          train_neg_items[start:end], train_neg_items_feature[start:end])
                        losses.append(loss.detach().cpu().item())

                        loss.backward(retain_graph=True)
                        self.anchor_optimizer.step()
                    step += 1
                    
                print('epoch training time: \t', (time.time() - s), ' sec')
                print('epoch training loss: \t', sum(losses) / len(losses))
                print('********************* training epoch end *********************')

                print('&&&&&&&&&&&&&&&&&&&& test epoch begin &&&&&&&&&&&&&&&&&&&&')
                self.anchor.eval()
                with torch.no_grad():
                    start_time = time.time()
                    map_test, mrr_test, p_test, r_test, f1_test, hr_test, ndcg_test = \
                        self.performance_eval(self.anchor,
                                              self.data.compute_user_items_dict,
                                              self.data.ground_truth_user_items_dict,
                                              self.data.compute_user_items_feature_dict)

                    end_time = time.time()
                    result.append([p_test, r_test, f1_test, hr_test, ndcg_test])
                    if f1_test > max_value:
                        max_value = f1_test
                        best_result_index = epoch
                        self.save()

                    print('current best performance: ' + str(result[best_result_index]))
                    print('epoch:', epoch, 'testing performance: ', p_test, r_test, f1_test, hr_test, ndcg_test,
                          'testing time: \t', (end_time - start_time), ' sec')
                    print('&&&&&&&&&&&&&&&&&&&& test epoch end &&&&&&&&&&&&&&&&&&&&')

            print('-------------------- learning end --------------------')
            print(result)
            print(best_result_index)
            print('anchor best performance: ' + str(result[best_result_index]))
            return result[best_result_index], result
        else:
            self.load()
            self.achieved  = 0
            print('&&&&&&&&&&&&&&&&&&&& original testing performance &&&&&&&&&&&&&&&&&&&&')
            self.anchor.eval()
            with torch.no_grad():
                start_time = time.time()
                target_map_test, target_mrr_test, target_p_test, target_r_test, target_f1_test, target_hr_test, target_ndcg_test = \
                    self.performance_eval(self.anchor,
                                          self.data.compute_user_items_dict,
                                          self.data.ground_truth_user_items_dict,
                                          self.data.compute_user_items_feature_dict)
                end_time = time.time()
                print('original testing performance: ', target_p_test, target_r_test, target_f1_test, target_hr_test, target_ndcg_test, 'testing time: \t', (end_time - start_time), ' sec')


            result = []
            max_value = 0
            best_result_index = 0
            
            gen_train_users, gen_train_users_feature, gen_train_pos_items, \
            gen_train_pos_items_feature, gen_train_neg_items, gen_train_neg_items_feature = self.counter_data_generation(self.args.anchor_model)

            train_users = self.data.user_all + gen_train_users
            train_users_feature = self.data.user_feature_all + gen_train_users_feature
            train_pos_items = self.data.pos_item_all + gen_train_pos_items
            train_pos_items_feature = self.data.pos_feature_all + gen_train_pos_items_feature
            train_neg_items = self.data.neg_item_all + gen_train_neg_items
            train_neg_items_feature = self.data.neg_feature_all + gen_train_neg_items_feature

            for name, param in self.anchor.named_parameters():
                param.requires_grad = True
                
            for epoch in range(self.args.epoch_number):
                print('********************* training epoch begin *********************')
                step = 0
                train_record_number = len(train_users)
                print('training sample number: ', train_record_number)

                train_users, train_users_feature, train_pos_items, train_pos_items_feature, train_neg_items, train_neg_items_feature = \
                    self.shuffle(train_users, train_users_feature,
                                 train_pos_items, train_pos_items_feature,
                                 train_neg_items, train_neg_items_feature)

                max_steps = train_record_number / self.args.batch_size
                s = time.time()
                losses = []
                while step <= max_steps:
                    if (step + 1) * self.args.batch_size > train_record_number:
                        b = train_record_number - step * self.args.batch_size
                    else:
                        b = self.args.batch_size
                    start = step * self.args.batch_size
                    end = start + b
                    if end > start:
                        self.anchor.train()
                        self.anchor_optimizer.zero_grad()
                        loss = self.anchor(train_users[start:end], train_users_feature[start:end],
                                          train_pos_items[start:end], train_pos_items_feature[start:end],
                                          train_neg_items[start:end], train_neg_items_feature[start:end])

                        losses.append(loss.detach().cpu().item())
                        loss.backward(retain_graph=True)
                        self.anchor_optimizer.step()
                    step += 1

                print('epoch training time: \t', (time.time() - s), ' sec')
                print('epoch training loss: \t', sum(losses) / len(losses))
                print('********************* training epoch end *********************')

                print('&&&&&&&&&&&&&&&&&&&& test epoch begin &&&&&&&&&&&&&&&&&&&&')
                warm_up_epoch = 5
                self.anchor.eval()
                with torch.no_grad():
                    start_time = time.time()
                    map_test, mrr_test, p_test, r_test, f1_test, hr_test, ndcg_test = \
                        self.performance_eval(self.anchor,
                                              self.data.compute_user_items_dict,
                                              self.data.ground_truth_user_items_dict,
                                              self.data.compute_user_items_feature_dict)

                    end_time = time.time()
                    result.append([p_test, r_test, f1_test, hr_test, ndcg_test])
                    if epoch > warm_up_epoch and f1_test > max_value:
                        max_value = f1_test
                        best_result_index = epoch

                    if epoch > warm_up_epoch and f1_test > target_f1_test:
                        self.achieved = 1

                    print('current best performance: ' + str(result[best_result_index]))
                    print('epoch:', epoch, 'testing performance: ', p_test, r_test, f1_test, hr_test, ndcg_test,
                          'testing time: \t', (end_time - start_time), ' sec')
                    print('&&&&&&&&&&&&&&&&&&&& test epoch end &&&&&&&&&&&&&&&&&&&&')

            print('-------------------- learning end --------------------')
            print(result)
            print(best_result_index)
            print('anchor best performance: ' + str(result[best_result_index]))
            print('final best performance: ' + str(result[best_result_index]), ':' + str(self.achieved))
            return result[best_result_index], result



    def performance_eval(self, model, compute_user_items_dict, ground_truth_user_items_dict, compute_user_items_feature_dict):
        #  pred { uid: {iid : score, }}
        #  ground_truth { uid: [tid...] }
        pred = dict()
        ground_truth = dict()
        for u in compute_user_items_dict.keys():
            items = compute_user_items_dict[u]
            features = compute_user_items_feature_dict[u]
            u_extend = [u] * len(items)
            u_feature_extend = [self.data.user_feature_attention[u]] * len(items)

            user_id_t = torch.LongTensor(u_extend).to(self.device)
            user_feature_t = torch.FloatTensor(u_feature_extend).to(self.device)
            item_id_t = torch.LongTensor(items).to(self.device)
            item_feature_t = torch.FloatTensor(features).to(self.device)

            scores = model.predict_score(user_id_t, user_feature_t, item_id_t, item_feature_t).cpu()
            pred[u] = dict(zip(items, scores))
            ground_truth[u] = ground_truth_user_items_dict[u]

            # if u == list(compute_user_items_dict.keys())[1]:
            #     index = np.argsort(np.array(scores))[::-1][:self.args.K]
            #     recommended = np.array(items)[index]
            #     print('recommendation case study: ')
            #     print('user ID: ' + str(u))
            #     print('recommended items: ' + str(recommended))
            #     print('recommended item scores: ' + str(scores))
            #     print('recommended item index: ' + str(index))
            #     print('recommended item index scores: ' + str(np.array(scores)[index]))
            #     print('real items in the testing set: ' + str(ground_truth[u]))

        map, mrr, p, r, f1, hit, ndcg = self.evaluator.evaluate(ground_truth, pred)
        return map, mrr, p, r, f1, hit, ndcg






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
    data.generate_pair_wise_training_corpus()
    #data.generate_point_wise_training_corpus()
    data.generate_validation_corpus()
    # (2) defining model
    
    if args.anchor_model == 4:
        user_len, dataset_len = data.statistics['user_number'], len(data.user_all)
        avg_number = int((dataset_len // user_len) * 1.0 * args.balanced_multiply)
        args.intervener_batch_size = avg_number * user_len - 10
    
    from config import anchor_type
    print ("Anchor Model Type is {}".format(anchor_type))
    if anchor_type == 'ele_mul': 
        from model.anchor_model import Anchor
    elif anchor_type == 'ele_add': 
        from model.anchor_ele_add import Anchor
    elif anchor_type == 'hybrid': 
        from model.anchor_hybrid import Anchor
    elif anchor_type == 'attention': 
        from model.anchor_attention import Anchor
    else : 
        raise Exception("anchor model is not valid, check the name")

    a_model = Anchor(args, data).to(device)
    i_model = Intervener(args, data).to(device)
    # (3) training the model based on the dataset.
    exp = experiment(args, a_model, i_model, data)

    result_1, _ = exp.run(mode = args.anchor_model)



