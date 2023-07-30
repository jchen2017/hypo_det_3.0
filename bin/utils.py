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

import argparse
import csv
import numpy as np
import re
import pickle
import sys

from features import load_topics, vectorize_by_topic, load_feats, load_comb_feats
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
load_existing_feat=1
load_small_wv=1 # 0: use large wv file  1: use small wv file

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-a', '--annotations', help='annotation input file')
	parser.add_argument('-t', '--train', help='message text training file location', nargs='+')
	parser.add_argument('--test', help='message text test file location', default=None)
	parser.add_argument('-k', '--knn', help='number of neighbors if model=knn (default 1)', default=1, type=int)
	parser.add_argument('-p', '--preprocess', help='bow, tfidf, top, tok, or emb?', default='tfidf')
	parser.add_argument('--rules', help='use rules to classify negative examples?', action='store_true')        
	parser.add_argument('-s', '--split', default=10, help='num splits', type=int)
	parser.add_argument('-m', '--model', help='[knn, dt, nb, nc]', default='knn')
	parser.add_argument('--param', help='', default='')
	parser.add_argument('--sampling', help='[down, up, down_test]', default='NA')
	parser.add_argument('--postprocessing', help='[r, nr]', default='nr')
	return parser.parse_args()

def load_data(annotations, message_files, preprocess):
	Xs = {}
	Ys = {}
	info_ls={}

	if load_existing_feat==0:
	#load messages
		for fname in message_files:
			print (fname)
			with open(fname, 'r') as infile:
				messages = infile.read().split('**********Message ID: ')
                        
				for m in messages:
					if m == '':
						continue
					#print ("===",m,"===")
					msgID = m[:m.find('\n')]
					msg_start = m.find('Subject:') 
					msg_end = m.find('------Original Message', msg_start)

                                
					msg = m[msg_start: msg_end]
					#handle illformat 
					if msg_start == -1:
						msg=re.sub(r'^\s*\d+\s*\n',"",m)
						print ("warning-1, special treatment: %s: %s"%(msgID, m))
                                        
					msgID=msgID.strip()
					msg=msg.strip()
					msg=re.sub("\\r\\n"," ",msg)
					Xs[msgID] = msg
                                
                                
                                
		print ("total %d instances"%(len(Xs.keys())))
		#load ys and related info
		with open(annotations, 'r') as infile:
			datareader = csv.reader(infile, delimiter='\t')
			next(datareader)
			for row in datareader:
				print (row)
		
				msgID=row[0]
				try:
					info=row[2]+"\t"+row[3]
				except:
					info="n/a"
				label=row[1]
                        
				if label == 'Y':
					Ys[msgID]=1
					info_ls[msgID]=info
				elif label == 'N':
					Ys[msgID]=0
				else:
					print ("illegal format of annotation file %s"%(annotations))

	X = []
	X_info = []
	Y = []
	ids = []
	Y_rules = []
	fn = 0
	fn_rm={}
	rules = 0
	for key, val in Xs.items():
		if key == '':
			continue
		if val == '':
			print ("empty value for key %s"%(key))
			continue
                
		X.append(val)
		X_info.append([val])
		ids.append(key)
		Y.append(Ys[key])
		Y_rules.append(get_rules_score(val))

		if get_rules_score(val) == 0:
			rules += 1
		if Ys[key.strip()] == 1 and get_rules_score(val) == 0:
			fn += 1
			fn_rm[key]=val
	print ("%d false negatives removed by rules"%(fn))
	for docid in fn_rm:
		print ("\t%s : %s"%(docid,fn_rm[docid]))
	print ("%d instances removed by rules"%(rules))


        #generate input for LDA
        '''
        f_idx=0
        outdir="/home/vhabedchenj/projects/EHR/hypo_det/expts/topicM/input_dir/test/"
        for fid in ids:
                outfile_lda=outdir+"%s.txt"%(fid)
                fout=open(outfile_lda, "w")
                X_cont=X[f_idx]
                X_cont=re.sub(r"^[sS]ubject:", "", X_cont)                
                fout.write(X_cont+"\n")
                fout.close()
                f_idx+=1

        exit (0)
        '''
   
	if preprocess == 'tfidf':
		if load_existing_feat==1:
			(X_vect, Y, Y_rules, ids, X_info)=load_feats("./feats/"+preprocess+"_feat_clean.pkl")
                        
		else:
			vect=TfidfVectorizer(stop_words='english')
			X_vect=vect.fit_transform(X)
                
			f=open("./feats/tfidf_feat.pkl", 'wb')
			pickle.dump(X_vect, f)
			pickle.dump(Y,f)
			pickle.dump(Y_rules,f)
			pickle.dump(ids,f)
			f.close()


	elif preprocess == 'bow':
		X_vect = CountVectorizer().fit_transform(X)
	elif preprocess == 'embs':  # word embeddings
		embs, w2i, i2w = load_word_embeddings()
		X_vect = vectorize(X, w2i, i2w, embs)
	elif preprocess == 'tok':  # return tokenized input, no other preprocessing
		X_vect = [word_tokenize(x.decode('utf-8').lower()) for x in X]
		X_vect = np.array(X_vect)
	elif preprocess == 'top':  # topic features
		if load_existing_feat==1:
			(X_vect, new_Y,new_Y_rules, new_ids, X_info)=load_feats("./feats/"+preprocess+"_feat.pkl")
		else:
			embs, w2i, i2w = load_word_embeddings_v2(load_small_wv,X)
			topic_dict=load_topics(embs,w2i)
			(X_vect,new_Y,new_Y_rules, new_ids, X_info) = vectorize_by_topic(X, Y, Y_rules, ids, w2i, i2w, embs, topic_dict)

		return X_vect, new_Y, new_Y_rules, new_ids, X_info

	elif re.search(r"_", preprocess):
                feat_ls=preprocess.split("_")
                print(feat_ls)
                if re.search(r"^top_lda\d+$", preprocess):
                        (X_vect, Y, Y_rules, ids, X_info)=load_comb_feats(feat_ls)
                else:
                        (X_vect, Y, Y_rules, ids, X_info)=load_feats("./feats/"+preprocess+"_feat_clean.pkl")

        elif re.search(r"lda\d+", preprocess):
                if load_existing_feat==1:
                        (X_vect, Y, Y_rules, ids, X_info)=load_feats("./feats/"+preprocess+"_feat.pkl")
                else:
                        (X_vect, Y, Y_rules, ids, X_info)=load_lda_feats()                
        
	else: # raw input
		X_vect = [ x.split(" ") for x in X]
		X_vect = np.array(X_vect)
		print ("raw text as input!")
        
	return X_vect, Y, Y_rules, ids, X_info


def vectorize(X, w2i, i2w, embs):
	result = []
	errors = 0
	parsed = 0
	for sent in X:
		tokens = word_tokenize(sent.decode('utf-8'))
		vectors = []
		for t in tokens:
			parsed += 1
			try:
				vectors.append(embs[w2i[t.lower()]])
			except:
				errors += 1
		avg = np.mean(vectors, axis=0)

		result.append(avg)
	print (errors)
	print (parsed)
	return np.vstack(result)


def score_x(txt):
	keywords = ['blood sugar', 'dizzy', 'dropped', 'dropping', 'sweat', 'lowest', 'stop', 'decrease', 'pass out']  # exclude: dose 'headache', 'hungry', 'sleepy', 'pale', 'shake', 'weak', 'drop', 'down','blur', 'confuse',
	keywords = ['sugar', 'glucose', 'cbg', 'metformin'] #, 'stop'] # 'dropped', 'pass out', 'passed out', 'blood sugar', 'glucose'] #, 
	keywords = ['glucose', 'sugar']	
	for k in keywords:
		if k in txt:
			return 1

	return 0

def score_x_2(txt):
	search = re.findall(r'\s+(\d{2})[\s:,?]+'," "+txt)
        
	times = ['pm', 'am', 'p.m.', 'a.m.']
	
	if search is not None and len(search) > 0:
		for s in search:
			if float(s) <= 85 and float(s) >= 35:
				return 1
			
	search = re.findall(r'\s+(\d{2})\.'," "+txt)
	if search is not None and len(search) > 0:
		for s in search:
			if float(s) <= 85 and float(s) >= 35:
				return 1
	search = re.findall(r'\s+(\d{2})mg', " "+txt)
	if search is not None and len(search) > 0:
		for s in search:
			if float(s) <= 85 and float(s) >= 35:
			#if float(s[:-2]) <= 85 and float(s[:-2]) >= 35:
                        	return 1
	
	search = re.findall(r"\s+(\d{2})\'", " "+txt)
	if search is not None and len(search) > 0:
		for s in search:
			if float(s) <= 85 and float(s) >= 35:
			#if float(s[:-1]) <= 85 and float(s[:-1]) >= 35:  # skip last char since it's always a single quote
				return 1	
	return 0


# simplified rules
def get_rules_score(txt):
        '''return -1 if we want the model to classify, 0 if we can rule it out via rules'''
        txt = " "+txt.lower()+" "
        txt = re.sub(r"[\n\r]", " ", txt)
        
        exact_match = [' incident', ' sugar', ' sugars', ' hypoglyc', ' glucose']

        for em in exact_match:
                if em in txt:
                        #print (em)
                        return -1
        return 0


def get_rules_score_v1(txt):
	'''return -1 if we want the model to classify, 0 if we can rule it out via rules'''
	txt = txt.lower()
	has_symptoms = 0
	symptoms = ['blur', 'confus', 'dizz', 'shaki', 'shake', 'sweat', 'weak', 'dose', 'drop', 'down']
	exact_match = ['low blood sugar', 'incident', 'low sugar', 'hypoglyc']

	for em in exact_match:
		if em in txt:
			#print (em)
			return -1
	for s in symptoms:
		if s in txt:
			has_symptoms += 1
	if score_x(txt) == 0:
		return 0
	if score_x_2(txt) == 0 and has_symptoms < 3:
		return 0

	return -1

## rule-based method
def get_rules_score_v2(txt):
	'''return -1 if we want the model to classify, 0 if we can rule it out via rul\es'''
	txt = txt.lower()
	has_symptoms = 0
	#symptoms = ['blur', 'confus', 'dizz', 'shaki', 'shake', 'sweat', 'weak', 'dose', 'drop', 'down']
	symptoms = [['blur'], ['confusion', 'confused'], ['dizzy', 'dizziness', 'lightheadedness', 'light headed', 'light-headed', 'light-headedness'], ['shaking', 'shake'], ['sweat', 'sweating'], ['weak'], ['hunger', 'hungry'], ['nervousness', 'anxiety'], ['sleepiness'], ['difficulty speaking'], ['loss of consciousness']]
        
	#exact_match = ['low blood sugar', 'incident', 'low sugar', 'hypoglyc']
        
	for em in exact_match:
		if em in txt:
			#print (em)
			return -1
	for sgrp in symptoms:
		has_sym=0
		for s in sgrp:
			if s in txt:
				has_sym=1
				break
		if has_sym == 1:
			has_symptoms += 1
        
	if score_x(txt) == 0:
		return 0
	#if score_x_2(txt) == 0 and has_symptoms < 3:
	if has_symptoms < 2:
		return 0

        return -1


def load_word_embeddings():
	'''build word embedding dictionaries word2ind and ind2word'''
	embeddings_loc = '/data/home1/wliu/Downloads/word2vec/hl7_umass_all_notes_word_vectors.txt'
	w2i = {}
	i2w = {}
	i = 0
	embs = []
	with open(embeddings_loc, 'r') as infile:
		for line in infile:
			ln = line.split(' ')
			key = ln[0]
			val = [float(j) for j in ln[1:]]
			if len(val) == 1:
				continue
			w2i[key] = i
			i2w[i] = key
			embs.append(val)
			i += 1
	embs = np.array(embs)
	return embs, w2i, i2w

### check: wv file; format treatment
def load_word_embeddings_v2(opt, X):
        '''build word embedding dictionaries word2ind and ind2word'''
        embeddings_loc = '/data/home1/wliu/Downloads/word2vec/hl7_umass_all_notes_word_vectors.txt'
        embeddings_hypo ='./word_embeddings/hl7_umass_all_notes_word_vectors_hypo.txt'
        #embeddings_loc = '/home/vhabedchenj/projects/EHR/hypoglycemia/word_embeddings/pubmed+wiki+pitts-nopunct-lower.tsv'
        #embeddings_hypo = './word_embeddings/pubmed+wiki+pitts-nopunct-lower_hypo.tsv'
        topics_loc = './domainKB/topics.txt'
        
        if opt == 0:
                voc={}
                for x in X:
                        words = word_tokenize(x.decode('utf-8'))
                        for wd in words:
                                wd=wd.lower()
                                voc[wd]=1
                               
                f=open(topics_loc, 'r')
                for line in f.readlines():
                        if re.search(r"^\d+\t",line):
                                (topid,content)=re.search(r"^(\d+)\t(.*)$",line).groups()
                                terms=content.split(',')
                                for term in terms:
                                        for wd in term.strip().split(" "):
                                                wd=wd.lower()
                                                voc[wd]=1
                f.close()

                w2i = {}
                i2w = {}
                i = 0
                embs = []
                infile = open(embeddings_loc, 'r')
                line=infile.readline()
                line=infile.readline()
                while line:
                        ln = line.strip().split(' ')
                        key = ln[0]
                        val = [float(j) for j in ln[1:]]
                        
                                                
                        if not key in voc or len(val) == 1:
                                line = infile.readline()
                                continue
                                
                        w2i[key] = i
                        i2w[i] = key
                        embs.append(val)
                        i += 1
                        line = infile.readline()

                infile.close()
                embs = np.array(embs)

                f=open(embeddings_hypo, 'wb')
                embs_norm=preprocessing.normalize(embs)
                pickle.dump(embs_norm,f)
                pickle.dump(w2i,f)
                pickle.dump(i2w,f)
                f.close()
                            
        elif opt == 1:
                f=open(embeddings_hypo, 'rb')
                if sys.version_info[0] < 3:
                        embs_norm=pickle.load(f)
                        w2i=pickle.load(f)
                        i2w=pickle.load(f)
                else:
                        embs_norm=pickle.load(f, encoding='latin1')
                        w2i=pickle.load(f, encoding='latin1')
                        i2w=pickle.load(f, encoding='latin1')
                                                                        
                f.close()
                                               
        return embs_norm, w2i, i2w


