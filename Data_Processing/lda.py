# coding: utf-8

# In[1]:

'''
LDA implement
input : lda_data.csv    5 col
output: lda_output.csv  1 col
'''
import csv
from gensim import corpora, models
import gensim


# In[2]:

def load_data(filename):
    table = []
    handle = open(filename)
    for lines in handle:
        line = lines.strip().replace(',',' ').split('\r')  # list of string
    for words in line:                                     # string
        words = words.lower()
        word = words.split()                               # list of string

        if word[0] == '0':
            word[0] = 'none'
        else:
            word[0] = 'injurd'
        if word[1] == '0':
            word[1] = 'none'
        else:
            word[1] = 'killed'

        table.append(word)
    handle.close()

    return table


# In[3]:

doc_set = load_data('lda_data.csv')
print 'There are', len(doc_set), 'accidents imported.'


# In[4]:

# stop words cleaning
def clean_data(dataset):
    stop_wd = ['none','unspecified','unknown','/','other']
    for k in range(len(doc_set)):
        doc = [i for i in dataset[k] if not i in stop_wd]
        dataset[k] = doc
    return dataset


# In[5]:

doc_set = clean_data(doc_set)


# In[6]:

def lda(dataset,num_tpc,num_pass):
    dictionary = corpora.Dictionary(dataset)
    corpus = [dictionary.doc2bow(doc) for doc in dataset]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = num_tpc, id2word = dictionary, passes = num_pass)
    topic_list = []
    for i in range(len(corpus)):
        prob = ldamodel[corpus[i]]  # list of tuple
        topic = max(prob,key=lambda item:item[1])[0]
        topic_list.append(topic)
    return ldamodel, topic_list


# In[7]:
[ldamodel, topic_list] = lda(doc_set, 3, 30)
print 'There are', len(topic_list), 'topics assigned.'


# In[8]:
def export_csv(filename,lst):
    myfile = open(filename, 'w')
    wr = csv.writer(myfile, delimiter='\n',quoting=csv.QUOTE_ALL)
    wr.writerow(lst)
    myfile.close()


# In[9]:

export_csv('lda_output.csv', topic_list)


# In[10]:

print(ldamodel.print_topics(num_topics=3, num_words=4))


# In[6]:

print(ldamodel.print_topics(num_topics=4, num_words=3))



# In[ ]:
