python softmax2.py -train ../dataset/dog_train -test ../dataset/dog_test -o pred.txt -alpha 0.001 -eta 0.01 -epsion 0.001 -iter 80
python cons.py pred.txt ../trans_data/valid.txt ../submit/dog.txt
python ../evaluate/metric_F1.py ../trans_data/valid.label ../submit/dog.txt
