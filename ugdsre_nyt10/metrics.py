import sklearn
import numpy as np

class metrics(object):
    def __init__(self, y_true, y_pred):
        self.y_pred = y_pred#[[0.34, 0.1, ..., 0.9], ...](nb_sample, nb_class)
        self.y_true = y_true

    def precision_at_k(self, lst_k):
        lst_tup_pred = []
        for sample_id, sample_prob in enumerate(self.y_pred):#[0.34, 0.1, ..., 0.9]
            cls_id = np.argmax(sample_prob)
            cls_prob = sample_prob[cls_id]

            tup = (sample_id, cls_id, cls_prob)
            lst_tup_pred.append(tup)

        lst_tup_pred = sorted(lst_tup_pred, key=lambda e: e[2], reverse=True)
        lst_sample_cls_pred = [(e[0], e[1]) for e in lst_tup_pred if e[1] != 0]

        #print len(lst_sample_cls_pred)
        #print len(lst_tup_pred)

        lst_sample_cls_true = []
        for sample_id, sample_true in enumerate(self.y_true):
            cls_id = np.argmax(sample_true)
            if cls_id != 0:
                lst_sample_cls_true.append((sample_id, cls_id))

        lst_p_at_k = []
        for k in lst_k:
            nb_correct = 0.0
            for sample_cls in lst_sample_cls_pred[:k]:
                if sample_cls in lst_sample_cls_true:
                    nb_correct += 1

            pr_at_k = nb_correct/k
            lst_p_at_k.append(pr_at_k)
            print 'P@%s: %s' % (k, pr_at_k)
        print 'mean: %s' % np.mean(lst_p_at_k)
        print 'numb_not_na_pred: %s' % len(lst_sample_cls_pred)
        
    def precision_recall(self, nb_step):
        lst_tup_pred = []
        for sample_id, sample_prob in enumerate(self.y_pred):#[0.34, 0.1, ..., 0.9]
            cls_id = np.argmax(sample_prob)
            cls_prob = sample_prob[cls_id]
            tup = (sample_id, cls_id, cls_prob)
            lst_tup_pred.append(tup)

        lst_tup_pred = sorted(lst_tup_pred, key=lambda e: e[2], reverse=True)
        lst_sample_cls_pred = [(e[0], e[1]) for e in lst_tup_pred if e[1] != 0]
        
        lst_sample_cls_true = []
        for sample_id, sample_true in enumerate(self.y_true):
            cls_id = np.argmax(sample_true)
            if cls_id != 0:
                lst_sample_cls_true.append((sample_id, cls_id))

        def cal_pre_rec(lst_sc_pred, lst_sc_true):
            nb_sc_pred = len(lst_sc_pred)
            nb_sc_true = len(lst_sc_true)
            
            nb_correct = 0.0
            for sc in lst_sc_pred:
                if sc in lst_sc_true:
                    nb_correct += 1
            pre1 = nb_correct/nb_sc_pred
            rec1 = nb_correct/nb_sc_true
            return pre1, rec1

        lst_pre_rec = []
        nb_sample = len(lst_sample_cls_pred)
        for rn in range(0, nb_sample, nb_step):
            sub_lst = lst_sample_cls_pred[:rn + nb_step]
            pre1, rec1 = cal_pre_rec(sub_lst, lst_sample_cls_true)
            lst_pre_rec.append([pre1, rec1])

        return np.array(lst_pre_rec)
