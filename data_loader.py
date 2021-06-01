import numpy as np
import pickle
import scipy.sparse as sp
import pandas as pd
import random



class DataLoader():
    def __init__(self, path):
        print("loading data ...")
        np.random.seed(0)
        random.seed(0)
        self.path = path
        self.statistics = dict()
        user_id_dict = pickle.load(open(self.path + "user_id_dict","rb"))
        item_id_dict = pickle.load(open(self.path + "item_id_dict", "rb"))
        feature_id_dict = pickle.load(open(self.path + "feature_id_dict", "rb"))
        self.statistics['user_number'] = len(user_id_dict.keys())
        self.statistics['item_number'] = len(item_id_dict.keys())
        self.statistics['feature_number'] = len(feature_id_dict.keys())


        self.train_batch_data = pd.read_csv(self.path + "train_data", header=None, dtype='str')
        self.user_feature_attention = pickle.load(open(self.path + "predicted_user_feature_attention", "rb"))
        self.item_feature_quality = pickle.load(open(self.path + "predicted_item_feature_quality", "rb"))

        self.train_user_positive_items_dict = pickle.load(open(self.path + "train_user_positive_items_dict", "rb"))
        self.train_user_negative_items_dict = pickle.load(open(self.path + "train_user_negative_items_dict", "rb"))

        self.ground_truth_user_items_dict = pickle.load(open(self.path + "test_ground_truth_user_items_dict", "rb"))
        self.compute_user_items_dict = pickle.load(open(self.path + "test_compute_user_items_dict", "rb"))

        print(self.statistics)

        self.user_all = []
        self.user_feature_all = []
        self.pos_item_all = []
        self.pos_feature_all = []
        self.neg_item_all = []
        self.neg_feature_all = []
        self.label_all = []


    def generate_pair_wise_training_corpus(self):
        for user, positive_items in self.train_user_positive_items_dict.items():
            if user in self.train_user_negative_items_dict.keys():
                for item in positive_items:
                    user_feature = self.user_feature_attention[user]
                    pos_item_id = int(item)
                    pos_item_feature = self.item_feature_quality[pos_item_id]
                    neg_item_id = random.choice(self.train_user_negative_items_dict[user])
                    neg_item_feature = self.item_feature_quality[neg_item_id]

                    self.user_all.append(user)
                    self.user_feature_all.append(user_feature)
                    self.pos_item_all.append(pos_item_id)
                    self.pos_feature_all.append(pos_item_feature)
                    self.neg_item_all.append(neg_item_id)
                    self.neg_feature_all.append(neg_item_feature)




    def generate_validation_corpus(self):
        self.compute_user_items_feature_dict = dict()
        for user, item_list in self.compute_user_items_dict.items():
            tmp = []
            for item in item_list:
                tmp.append(self.item_feature_quality[item])
            self.compute_user_items_feature_dict[user] = tmp


