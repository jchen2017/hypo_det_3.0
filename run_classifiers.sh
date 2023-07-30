#!/bin/sh
#

dir=./
logdir=./logfiles/
ann_dir=./annotations/

for model in logit linearsvc
do
    for feat in top tfidf tfidf_top
    do
	python $dir/bin/hypoglycemia_api.py -a $ann_dir/annotations_complete.txt -t $ann_dir/randomly_selected_3000.txt  -p $feat -m $model --rules -s 10 > $logdir/cv_${model}_${feat}_rule_v2.log
	
	
    done
done


for model in logit linearsvc rf dt nc knn knn3u knn7u knn5u knn11u
do
    for feat in tfidf tfidf_top tfidf_lda100 top_lda100 tfidf_top_lda100 tfidf_top tfidf_lda100 top_lda100 top tfidf lda100
    do
	for spl in ns up.smote.r up.smotesvm.r up.smoteb1.r down.allknn.r down.nm3.r down.enn.r down.cc.r up.smoteenn.r up.smotetomek.r #up.smoteb2.r up.smotesvm.r up.adasyn.r down.tomek.r down.iht.r down.enn.r #ns down.rule.r up.smote.r up.ros.r down.rus.r up.smoteb1.r up.smoteb2.r up.smotesvm.r down.iht.r down.allknn.r down.enn.r down.cc.r up.smoteenn.r up.smotetomek.r down.rus.nr
	do
	    for pp in "nr" "r"
	    do		    
		echo "python $dir/bin/hypoglycemia_api.py -a $ann_dir/annotations_complete.txt -t $ann_dir/randomly_selected_3000.txt -p $feat -m $model --sampling $spl --postprocessing $pp"
		python $dir/bin/hypoglycemia_api.py -a $ann_dir/annotations_complete.txt -t $ann_dir/randomly_selected_3000.txt -p $feat -m $model --sampling $spl --postprocess $pp > $logdir/cv_${model}_${feat}_${spl}_pp${pp}.log &
	    done
        done
    done
done
