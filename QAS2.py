import glob
import nltk
import numpy as np
import random as rnd


path ="/Users/Yosafat VS/Documents/PBA_Project/Textes"

def array_data_gather(path):
    text = []
    txt_files = glob.glob(path + "/*.txt")
    for gw in range(len(txt_files)):
        files = open(txt_files[gw])
        text.append(files.read().replace("\n", " "))
        gw + +1
    return text
def tokenized2(text):
    tokens = nltk.pos_tag(nltk.word_tokenize(text))
    return tokens

def tokenized(text):
    STOP_TYPES = ['\/','``',':','POS','RP','PRP$','(',')','TO','DT', 'CC' , 'RB' , 'PRP' ,'IN',',','.','?']
    tokens = nltk.pos_tag(nltk.word_tokenize(text))
    good_words = [w for w, wtype in tokens if wtype not in STOP_TYPES]
    return good_words

def get_question():
    array_of_question=["Hello , can i help you?","Good day!\nMay I help you?","Ohayou Onii-chan! Can i Help You?"]
    num=rnd.randint(0,(len(array_of_question)-1))
    print(array_of_question[num])
    question = input()
    return question

def get_context(question):
    context=""
    qt=question.lower()
    questpack = nltk.word_tokenize(qt)
    if('who' in questpack):
        context="PERSON"
    if('when' in  questpack):
        context="DATE"
    if('prize' in questpack):
        context="MONEY"
    if ('fee' in questpack):
        context = "MONEY"
    if('information' in  questpack):
        context="WEB"
    return context

def tf(question):
    data = array_data_gather(path)
    t1 = tokenized(data[0])
    t2 = tokenized(data[1])
    t3 = tokenized(data[2])
    t4 = tokenized(question)
    td = set(t1).union(set(t2)).union(set(t3)).union(set(t4))
    word_A = dict.fromkeys(td, 0)
    word_B = dict.fromkeys(td, 0)
    word_C = dict.fromkeys(td, 0)
    word_QS = dict.fromkeys(td, 0)

    for word in t1:
        word_A[word] += 1
    for word in t2:
        word_B[word] += 1
    for word in t3:
        word_C[word] += 1
    for word in t4:
        word_QS[word] += 1
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    for w1 in word_A.values():
        if(w1!=0):
            a1.append(float(1+np.log(w1)))
        else:
            a1.append(0)
    for w2 in word_B.values():
        if(w2!=0):
            a2.append(float(1+np.log(w2)))
        else:
            a2.append(0)
    for w3 in word_C.values():
        if(w3!=0):
            a3.append(float(1+np.log(w3)))
        else:
            a3.append(0)
    for w4 in word_QS.values():
        if(w4!=0):
            a4.append(float(1+np.log(w4)))
        else:
            a4.append(0)
    ad = []
    ad.append(a1)
    ad.append(a2)
    ad.append(a3)
    ad.append(a4)
    ad = np.asarray(ad)
    return ad

def idf(question):
    data = array_data_gather(path)
    t1 = tokenized(data[0])
    t2 = tokenized(data[1])
    t3 = tokenized(data[2])
    t4 = tokenized(question)
    ta = [t1,t2,t3,t4]
    v=len(ta)
    td = set(t1).union(set(t2)).union(set(t3)).union(set(t4))
    w1 = list(set(t1))
    w2 = list(set(t2))
    w3 = list(set(t3))
    w4 = list(set(t4))
    ta = dict.fromkeys(td,0)
    for word in w1:
        ta[word]+=1
    for word in w2:
        ta[word]+=1
    for word in w3:
        ta[word]+=1
    for word in w4:
        ta[word]+=1
    idfList=[]
    for wx in ta.values():
        idfList.append(float(np.log(v/wx)))
    idfList = np.asarray(idfList)
    return idfList

def tf_idf(tf,idf):
    for g in range(len(tf)):
        for h in range(len(tf[g])):
            tf[g][h]=tf[g][h]*idf[g]
    return tf

def cosinesim(matrix , vector):
    neighbors = []
    for row in range(matrix.shape[0]):
        vector_norm = np.linalg.norm(vector)
        row_norm = np.linalg.norm(matrix[row, :])
        cos_val = vector.dot(matrix[row, :]) / (vector_norm * row_norm)
        neighbors.append(cos_val)
    return neighbors

def tagginger(text):
    texte=[]
    text_list = nltk.pos_tag(text)
    for k in range(len(text_list)):
       texte.append([text_list[k][0],text_list[k][1]])
    return texte

def stopword(txt):
    from nltk.corpus import stopwords
    text2 = txt.lower()
    txt2 = nltk.word_tokenize(text2)
    nltk_words = list(stopwords.words('english'))  # About 150 stopwords
    output = [w for w in txt2 if not w in nltk_words]
    return output

def get_answer(anspack,text,question):
    answer=""
    indx=0
    filtered_text = stopword(text)
    filtered_question = stopword(question)
    text_list = tagginger(filtered_text)
    questions = tagginger(filtered_question)
    print(questions)
    if(anspack=='MONEY'):
        for w in range (len(text_list)):
            if(text_list[w][0]==questions[0][0] and text_list[w][1]=='CD'):
                    indx=w
    if(text_list[indx+1][0]=='prize'):
        status= 'true'
        if(status=='true'):
            answer=(text_list[indx+3][0])

    return answer

def analyze(qst):
    tfI = tf(qst)
    idfI = idf(qst)
    tdf = tf_idf(tfI, idfI)
    vct4 = tdf[3]
    csm_as=cosinesim(tdf, vct4)
    csm_as2=[csm_as[0],csm_as[1],csm_as[2]]
    csm_as2=np.asarray(csm_as2)
    min = np.argmax(csm_as2)
    data = array_data_gather(path)
    text=data[min]
    tfdf = open(r"C:\Users\Yosafat VS\Documents\PBA_Project\Textes\COSIMMERS\tf-idf_TABLE.txt", "w")
    tfdf.write(str(tdf))
    cosimmers = open(r"C:\Users\Yosafat VS\Documents\PBA_Project\Textes\COSIMMERS\Cosim_result.txt", "w")
    cosimmers.write(str(csm_as))
    return text

def mainQA():
    qst = get_question()
    text = analyze(qst)
    anspack = get_context(qst)
    print(anspack)
    textar = get_answer(anspack, text,qst)
    print(textar)

def main():
    ans=""
    import time
    start_time = time.time()
    mainQA()
    array_of_question = ["Anything else sir?", "Is there something you wanna ask me again?",
                             "Onni-chan , do you have some question again?"]
    num = rnd.randint(0, (len(array_of_question) - 1))
    print(array_of_question[num]+"\n")
    print("YES/NO")
    ans=input()
    if(ans=="YES"):
        mainQA()
    else:
        print("thank you and see you next time...")
    print("--- %s seconds ---" % (time.time() - start_time))

main()