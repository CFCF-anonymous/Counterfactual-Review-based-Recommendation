import numpy as np
import torch

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    '''
    [scipy.sparse] sparse_mx: col, row, data, shape, ...
    [torch.sparse.FloatTensor] indices, values, shape, ...
    '''
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



class Evaluate(object):
    def __init__(self, topk):
        self.Top_K = topk

    def MAP(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.Top_K]
            tmp = 0
            hit = 0
            for j in range(len(fit)):
                if fit[j] in v:
                    hit += 1
                    tmp += hit / (j + 1)
            result.append(tmp)
        return np.array(result).mean()

    def MRR(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.Top_K]
            tmp = 0
            for j in range(len(fit)):
                if fit[j] in v:
                    tmp = 1 / (j + 1)
                    break
            result.append(tmp)
        return np.array(result).mean()

    def NDCG(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.Top_K]
            temp = 0
            Z_u = 0

            for j in range(min(len(fit), len(v))):
                Z_u = Z_u + 1 / np.log2(j + 2)
            for j in range(len(fit)):
                if fit[j] in v:
                    temp = temp + 1 / np.log2(j + 2)

            if Z_u == 0:
                temp = 0
            else:
                temp = temp / Z_u
            result.append(temp)
        return np.array(result).mean()

    def top_k(self, ground_truth, pred):
        p_total = []
        r_total = []
        f_total = []
        hit_total = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.Top_K]
            cross = float(len([i for i in fit if i in v]))
            p = cross / len(fit)
            r = cross / len(v)
            if cross > 0:
                f = 2.0 * p * r / (p + r)
            else:
                f = 0.0
            hit = 1.0 if cross > 0 else 0.0
            p_total.append(p)
            r_total.append(r)
            f_total.append(f)
            hit_total.append(hit)
        return np.array(p_total).mean(), np.array(r_total).mean(), np.array(f_total).mean(), np.array(hit_total).mean()

    def evaluate(self, ground_truth, pred):
        # pred { uid: {iid : score, }}
        # ground_truth { uid: [tid...] }
        # map, mrr, p, r, f1, hit, ndcg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        sorted_pred = {}
        for k, v in pred.items():
            sorted_pred[k] = sorted(v.items(), key=lambda item: item[1])[::-1]

        # Protocol : pred { uid: [(tid, score)...)] }
        # Protocol : ground_truth { uid: [tid...] }
        p, r, f1, hit = self.top_k(ground_truth, sorted_pred)
        map = self.MAP(ground_truth, sorted_pred)
        mrr = self.MRR(ground_truth, sorted_pred)
        ndcg = self.NDCG(ground_truth, sorted_pred)
        return map, mrr, p, r, f1, hit, ndcg


if __name__ == '__main__':
    e = Evaluate(3)
    pred = {1:{1:0.4, 2:0.43, 3:1.2, 4:0.81, 5:0.7, 6:0.32}, 2:{1:0.2, 2:0.93, 3:0.11, 4:0.43, 5:0.27, 6:0.12}}
    ground_truth = {1:[3], 2:[1,2]}
    print(e.evaluate(ground_truth, pred))