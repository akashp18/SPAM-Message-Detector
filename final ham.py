from flask import Flask,render_template,request,url_for
from flask_script import Manager
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix

labels=[]
contents=[]
emailLabels=[]
app=Flask(__name__)
manager =Manager(app)
"""Split the contents and labels"""
with open('SMSSpamCollection','r') as f:
	filer=f.readlines()
	for line in filer:
		labels.append(line[:4].strip())
		contents.append(line[4:].strip())

"""Split train test data in ratio of 80:20"""
labels_train,labels_test,contents_train,contents_test=train_test_split(labels,contents,test_size=0.2,random_state=42)


""" Create dictionary containing the frequency of each word in the data """
count=CountVectorizer()
count_cross=CountVectorizer()

sms_transform=count.fit_transform(contents_train)
sms_test=count.transform(contents_test)
print(len(count.vocabulary_))

#cross_val_score_transforamtion
sms_transform_cross=count_cross.fit_transform(contents)

""" Create a document matrix of each word in each document """
features=TfidfTransformer()
features_cross=TfidfTransformer()

feat=features.fit_transform(sms_transform)
contents1=features.transform(sms_test).toarray()
print(feat.shape)
#TfidfTransformer_cross_val_transform
feat_cross=features_cross.fit_transform(sms_transform_cross)

"""Create SVM classifier"""
clf=svm.LinearSVC(C=10000.0)#,kernel="rbf",gamma=0.6)
clf.fit(feat,labels_train)
pred=clf.predict(contents1)

#cross_val_score_svm

clf_cross_val=svm.LinearSVC(C=10000.0)
cross_matrix = cross_val_score(clf_cross_val, feat_cross, labels)
print("Accuracy Score by Cross Validation: %s"%cross_matrix)

acc=accuracy_score(labels_test,pred)

print ("Accuracy score by train_test_split: %s"%acc)
print("Average Accuracy score of Cross Validation %s"%(sum(cross_matrix)/len(cross_matrix)))

"""Confusin Matrix"""

print(confusion_matrix(pred, labels_test))
#C[0,0] contains True negatives i.e. True Not Spam detected as Not Spam
#C[0,1] contains False Positives i.e. True Not Spam detected as Spam
#C[1,0] contains False Negatives i.e. True Spam detected as Not Spam
#C[1,1] Contains False Positives i.e. True Spam deteceted as Spam

#@app.route('/spam/<text>')
def give_output(text):
	ml=text
	chk=count.transform([ml])
	trans=features.transform(chk)
	prediction=str(clf.predict(trans))
	print (type(prediction))
	return prediction

@app.route('/spam',methods=['GET','POST'])
def take_input():
	if request.method=='POST':
		test=(request.form['message'])
		test_doc=give_output('hi')
		return render_template('final.html',test=test_doc)
	return render_template('input.html')



if __name__=='__main__':
	manager.run()
