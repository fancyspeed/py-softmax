#coding: utf-8
import sys
import argparse
import random
import math

class Softmax(object):
    def __init__(self):
        self.label_dict = {}
        self.label_num = 0
        self.max_word_idx = 0
        self.doc_num = 0
        
    def add_label(self, label):
        if label not in self.label_dict:
            self.label_dict[label] = [] 

    def init_model(self, p_train):
        for line in open(p_train):
            if not line.strip() or line[0] == '#':
                continue
            row = line.rstrip().split(' ')
            labels = row[0].split(',')
            for label in labels:
                if label: 
                    self.add_label(label)
            feat_pairs = row[1:]
            for pair in feat_pairs:
                word_idx, word_count = pair.split(':') 
                self.max_word_idx = max(int(word_idx), self.max_word_idx)
            self.doc_num += 1
        self.label_num = len(self.label_dict)
        for label in self.label_dict:
            for word in range(self.max_word_idx+1):
                self.label_dict[label].append( random.random() * 0.01 / (self.max_word_idx+1) )

        print 'doc num', self.doc_num
        print 'label num', self.label_num
        print 'max feat idx', self.max_word_idx

    def load_line(self, line):
        labels, feats = {}, {} 
        if not line.strip() or line[0] == '#':
            return labels, feats

        row = line.rstrip().split(' ')
        if row[0].find(':') < 0:
            labels = {v:1 for v in row[0].split(',') if v}
        else:
            labels = {v.split(':')[0]:float(v.split(':')[1]) for v in row[0].split(',') if v}

        feat_pairs = row[1:]
        for pair in feat_pairs:
            word_idx, word_count = pair.split(':') 
            feats[int(word_idx)] = float(word_count)
        return labels, feats

    def predict(self, feats):
        labels = {}
        tot = 0
        for label in self.label_dict:
            labels[label] = 0
            for word_idx in feats:
                labels[label] += feats[word_idx] * self.label_dict[label][word_idx]
            labels[label] = math.exp(labels[label])
            tot += labels[label]
        if tot <= 0: return labels
        for label in labels:
            labels[label] /= tot
        return labels 

    def gradent_descent(self, labels, feats, alpha=0.001, eta=0.01):
        predicts = self.predict(feats)
        loss = 0
        for label in labels:
            grad = - labels[label] * (1 - predicts[label])
            for word_idx in feats:
                update = grad * feats[word_idx] + eta * self.label_dict[label][word_idx]
                self.label_dict[label][word_idx] -= alpha * update 
            loss += - predicts[label] * labels[label]
        return loss 

    def get_error(self, labels, feats, predicts={}):
        loss = 0 
        if not predicts:
            predicts = self.predict(feats)
        for label in labels:
            loss += - predicts[label] * labels[label]
        return loss 

    def train(self, p_train, alpha=0.001, eta=0.01, epsion=0.0001, max_iter=50):
        last_rmse, this_rmse = self.doc_num * 10000, self.doc_num * 10000
        for iter in range(max_iter):
            train_rmse = 0
            for line in open(p_train): 
                labels, feats = self.load_line(line)
                error = self.gradent_descent(labels, feats, alpha, eta)
                train_rmse += error 
            this_rmse = 0
            for line in open(p_train): 
                labels, feats = self.load_line(line)
                error = self.get_error(labels, feats)
                this_rmse += error 
            print 'train loss:%s, this loss:%s' % (train_rmse, this_rmse)
            if last_rmse - this_rmse <= epsion:
                break
            last_rmse = this_rmse


    def infer(self, p_test, p_out):
        fo = open(p_out, 'w')
        for line in open(p_test): 
            labels, feats = self.load_line(line)
            p_dict = self.predict(feats)
            sort_list = sorted(p_dict.items(), key=lambda d:-d[1])
            str_list = ['%s:%s' % (k, v) for k, v in sort_list]
            fo.write(' '.join(str_list) + '\n')
        fo.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', required=True)
    parser.add_argument('-test', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-alpha',  type=float, default=0.001)
    parser.add_argument('-eta',  type=float, default=0.01)
    parser.add_argument('-epsion',  type=float, default=0.0001)
    parser.add_argument('-iter', dest='max_iter', type=int, default=50, help='total iteration')
    args = parser.parse_args()

    softmax = Softmax()
    softmax.init_model(args.train)
    softmax.train(args.train, args.alpha, args.eta, args.epsion, args.max_iter)
    softmax.infer(args.test, args.output)

