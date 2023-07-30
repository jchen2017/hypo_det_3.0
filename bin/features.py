# coding=utf-8
#
# Released under MIT License
#
# Copyright (c) 2019, Jinying Chen
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

import numpy as np
import re
import pickle
import sys

from sklearn import preprocessing

from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.parse.stanford import StanfordDependencyParser

def load_topics(embs,w2i):
        topic_dict={}
        topics_loc = './domainKB/topics.txt'
        f=open(topics_loc, 'r')
        for line in f.readlines():
                if re.search(r"^\d+\t",line):
                        (topid,content)=re.search(r"^(\d+)\t(.*)$",line).groups()
                        topid=int(topid)
                        topic_dict[topid]={}
                        topic_dict[topid]['terms']={}
                        topic_dict[topid]['vec']=np.zeros(len(embs[0]))
                        term_ls=content.split(",")
                        term_num=0
                        for term in term_ls:
                                term=term.strip().lower()
                                if term == "":
                                        continue
                                topic_dict[topid]['terms'][term]=1
                                try:
                                        term_vec=np.array(embs[w2i[term]])
                                        topic_dict[topid]['vec']+=term_vec
                                        topic_dict[topid]['terms'][term]=term_vec
                                        term_num+=1
                                except:
                                        pass
                        topic_dict[topid]['vec']=topic_dict[topid]['vec']/term_num
                        
        f.close()
        return topic_dict

def load_feats (featfile):
	print ("load features from %s"%(featfile))
	f=open(featfile, 'rb')
	if sys.version_info[0] < 3:
	        result=pickle.load(f)
	        labels=pickle.load(f)
	        rules=pickle.load(f)
	        docid_ls=pickle.load(f)
	else:
	        result=pickle.load(f,encoding='latin1')
	        labels=pickle.load(f,encoding='latin1')
	        rules=pickle.load(f,encoding='latin1')
	        docid_ls=pickle.load(f,encoding='latin1')

	f.close()
	result_info=[""]*len(labels)
	return (result,labels, rules, docid_ls, np.array(result_info))


def vectorize_by_topic(X, Y, Y_rules, ids, w2i, i2w, embs, topic_dict):
        win_size=2
        result = []
        result_info=[]
        labels = []
        rules=[]
        docid_ls = []
        has_keysent = []

        errors = 0
        parsed = 0
        topic_ls=topic_dict.keys()
        topic_num=len(topic_ls)
        feat_num=topic_num*2+5
        doc_id=0
        for doc in X:
                # debug
                #if doc_id > 10:
                #        exit (0)

                docid=ids[doc_id]
                print ("***** doc %s *****"%(docid))
                
                doc_vec=[]
                key_sent=[]
                sent_idx=0
                sent_ls=sent_tokenize(doc.decode('utf-8'))
                for sent in sent_ls:
                        tokens = word_tokenize(sent) #sent.decode('utf-8'))
                        postags = pos_tag(tokens)

                        sent_vec = []
                        i=0
                        for t in tokens:
                                #feat_vec=np.zeros(feat_num)

                                feat_vec=[0]*feat_num
                                t=t.lower()

                                tag=postags[i][1]
                                i+=1
                                                                
                                #focus
                                if t=="i" or t=="my":
                                        feat_vec[topic_num*2+4]=1

                                isNVJ=0
                                isIN=0
                                # check pos
                                if re.search(r"^[NVJ]", tag):
                                        isNVJ=1
                                elif tag=="IN":
                                        isIN=1

                                # check if t is a negation term
                                isNeg=0
                                if re.search(r'^n[o\']t$', t):
                                        isNeg=1
                                        feat_vec[topic_num*2+3]=1
                                        
                                # check if t is a number
                                isInt=0
                                try:
                                        t=int(t)
                                        feat_vec[topic_num*2+1]=1
                                        if t <= 70:
                                                feat_vec[topic_num*2+2]=1
                                        isInt=1
                                except:
                                        pass
                                
                                # check t's topic
                                if isNVJ == 1 or isIN == 1:
                                        has_topic=0
                                        for topid in topic_ls:
                                                if t in topic_dict[topid]['terms']:
                                                        feat_vec[topid]=1
                                                        has_topic=1
                                        if has_topic == 1:
                                                feat_vec[topic_num*2]=1
                                        else:
                                                feat_vec[topic_num*2]=-1

                                        if isNVJ==1 or has_topic == 1:
                                                parsed +=1
                                                try:
                                                        wv=embs[w2i[t.lower()]]
                                                        for topid in topic_ls:
                                                                feat_vec[topic_num+topid]=np.dot(topic_dict[topid]['vec'],wv)
                                                except:
                                                        errors+=1
                                
                                sent_vec.append(feat_vec)
                                
                        #sent level feat
                        if re.search(r"(hypoglyc)|(glucose)", sent) or re.search(r" sugar", sent) and re.search(r" ((low)|(below))", sent) or re.search(r"(blood sugar)|(sugar level)", sent):
                                key_sent.append(sent_idx)
                        
                        doc_vec.append(sent_vec)
                        sent_idx+=1

                        
                label=Y[doc_id]
                rule=Y_rules[doc_id]

                key_sent_id=-1
                if len(key_sent) > 1:
                        print("multiple key sentences for doc %s!"%(docid))
                        for sentid in key_sent:
                                if np.max(doc_vec[sentid],0)[-1] == 1:
                                        key_sent_id=sentid
                                        print("selection 1: pick key sent %d: %s"%(sentid, sent_ls[sentid].encode('utf-8','replace')))
                                        break
                        if key_sent_id == -1:
                                key_sent_id = key_sent[0]
                                print("selection 2: pick key sent %d: %s"%(sentid, sent_ls[sentid].encode('utf-8','replace')))
                elif len(key_sent) == 1:
                        key_sent_id = key_sent[0]

                if key_sent_id != -1:
                        sentid=key_sent_id
                        result.append([])
                        inst_vec=result[-1]
                        result_info.append([])
                        inst_info=result_info[-1]
                        inst_info.append(doc+"###")
                        has_keysent.append(1)

                        sent_st_id=max(sentid-win_size,0)
                        sent_end_id=min(sentid+win_size+1,len(doc_vec))
                        for i in range(sent_st_id,sent_end_id):
                                inst_info.append(sent_ls[i])
                                for word_feat_vec in doc_vec[i]:
                                        inst_vec.append(word_feat_vec)
                        labels.append(label)
                        rules.append(rule)
                        docid_ls.append(docid)
                else:
                        print("no key sentences!")
                        has_keysent.append(-1)
                        result.append([])
                        inst_vec=result[-1]
                        result_info.append([])
                        inst_info=result_info[-1]
                        inst_info.append(doc+"###")
                        sent_st_id=0
                        sent_end_id=min(2*win_size+1,len(doc_vec))
                        for i in range(sent_st_id,sent_end_id):
                                inst_info.append(sent_ls[i])
                                for word_feat_vec in doc_vec[i]:
                                        inst_vec.append(word_feat_vec)
                        labels.append(label)
                        rules.append(rule)
                        docid_ls.append(docid)

                doc_id+=1

  
        i=0
        for docid in docid_ls:
                
                rule=rules[i]
                label=labels[i]
                inst_text=result_info[i]
                vect=result[i]
                has_key=has_keysent[i]

                # collapse doc vectors
                vect_new=np.append(np.max(vect,0),0)
                
                if has_key:
                        vect_new[-1]=1
                                
                result[i]=vect_new
                i+=1

        result=np.array(result)
        part1_mtx=preprocessing.normalize(result[:,0:topic_num])
        top_mtx=preprocessing.normalize(result[:,topic_num:topic_num*2])
        part2_mtx=result[:,topic_num:len(result[0])]
        result=np.concatenate((part1_mtx,top_mtx,part2_mtx),axis=1)

        f=open("./feats/top_feat.pkl", 'wb')
        pickle.dump(result,f)
        pickle.dump(labels,f)
        pickle.dump(rules,f)
        pickle.dump(docid_ls,f)
        f.close()
                                        
        return (result,labels, rules, docid_ls, np.array(result_info))


def load_comb_feats (feat_names):
        i=0
        for feat in feat_names:
                if feat == "tfidf":
                        featfile="./feats/%s_feat_clean.pkl"%(feat)
                else:
                        featfile="./feats/%s_feat.pkl"%(feat)
                f=open(featfile, 'rb')
                if sys.version_info[0] < 3:
                        result1=pickle.load(f)
                else:
                        result1=pickle.load(f,encoding='latin1')

                        
                if feat == "tfidf":
                        result1=np.array(result1.toarray())
                if i == 0:
                        result=result1
                        if sys.version_info[0] < 3:
                                labels=pickle.load(f)
                                rules=pickle.load(f)
                                docid_ls=pickle.load(f)
                        else:
                                labels=pickle.load(f,encoding='latin1')
                                rules=pickle.load(f,encoding='latin1')
                                docid_ls=pickle.load(f,encoding='latin1')

                        result_info=[""]*len(labels)
                else:
                        result=np.concatenate((result,result1),axis=1)
                        if sys.version_info[0] < 3:
                                labels1=pickle.load(f)
                        else:
                                labels1=pickle.load(f,encoding='latin1')
                                
                        if len(labels1) != len(labels):
                                print ("mismatch of instances for different feature types")
                                exit (1)
                        
                i+=1
                f.close()
                 
        return (result, labels, rules, docid_ls, np.array(result_info))
