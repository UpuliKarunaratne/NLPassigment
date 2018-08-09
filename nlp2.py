import nltk
import pandas as pd
import string
import re
from nltk import bigrams
from nltk.probability import FreqDist


hamList = []
spamList1 = []
hamUnigrams = []
spamList = []
hamBigrams = []
hamBi = []
spamUnigrams = []
spamBigrams = []
hamUnigramsWithoutStop = []
input1Unigram = []
input2Unigram = []
input1Bigram = []
input2Bigram = []

fullCorpus = pd.read_csv('SMSSpamCollection.tsv',sep='\t', header=None)
fullCorpus.columns = ['Lable','MessageBody']
fullCorpus.head()

def punctuation_remove(rawMsg):
    nopuncMsg = "".join([char for char in rawMsg if char not in string.punctuation])
    nopuncMsg = "<s> "+nopuncMsg+" </s>"
    return nopuncMsg
fullCorpus["CleanedMessageBody"] = fullCorpus['MessageBody'].apply(lambda x: punctuation_remove(x.lower()))


def tokenize(noPuncMsg):
    tokens = re.split('\s+',noPuncMsg)
    return tokens
fullCorpus["TokenizedMessageBody"] = fullCorpus["CleanedMessageBody"].apply(lambda a: tokenize(a))


from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
def lemmatization(msgCorpus):
    lemmatized = [wnl.lemmatize(word)for word in msgCorpus]
    return lemmatized
fullCorpus["LemmatizedMessageBody"] = fullCorpus["TokenizedMessageBody"].apply(lambda d: lemmatization(d))


#Group HAM and Spam Messages 
for index, row in fullCorpus.iterrows():
    if (row["Lable"] == "ham"):
        hamList.append(row["LemmatizedMessageBody"])
    elif (row["Lable"] == "spam"):
        spamList1.append(row["LemmatizedMessageBody"])
        
#Convert to 1D array
hamList1d = [a for sub in hamList for a in sub] 
spamList = [a for sub in spamList1 for a in sub] 

hamBigrams.extend(nltk.bigrams(hamList1d))
spamBigrams.extend(nltk.bigrams(spamList))

stopWords = nltk.corpus.stopwords.words('english')
filteredBiHam = []
filteredBiSpam = []
for bigramH in hamBigrams:
    if bigramH[0] in stopWords and bigramH[1] in stopWords:
        continue
    filteredBiHam.append(bigramH)
    
for bigramS in spamBigrams:
    if bigramS[0] in stopWords and bigramS[1] in stopWords:
        continue
    filteredBiSpam.append(bigramS)
    

hamUnigrams = [word for word in hamList if word not in stopWords]
spamUnigrams = [word for word in spamList if word not in stopWords]

hamUnigrams = [a for sub in hamUnigrams for a in sub]
#spamUnigrams = [a for sub in spamUnigrams for a in sub]

inputS1 =  " Sorry, ..use your brain dear"
inputS2 =  " SIX chances to win CASH."

#Preproecss Sentences

inputS1 = inputS1.lstrip(' ') 
inputS2 = inputS2.lstrip(' ')

inputS1NoPun = "".join(char for char in inputS1 if char not in string.punctuation).lower()
inputS2NoPun = "".join(char for char in inputS2 if char not in string.punctuation).lower()

inputS1NoPun = "<s> "+ inputS1NoPun +" </s>" 
inputS2NoPun = "<s> "+ inputS2NoPun +" </s>" 

inputS1Tokenized = re.split('\s+',inputS1NoPun)
inputS2Tokenized = re.split('\s+',inputS2NoPun)

inputS1Lemmatized = [wnl.lemmatize(word)for word in inputS1Tokenized]
inputS2Lemmatized = [wnl.lemmatize(word)for word in inputS2Tokenized]


#Assign Unigrams and Bigrams

input1Unigram = inputS1Lemmatized
input2Unigram = inputS2Lemmatized

input1Bigram.extend(nltk.bigrams(input1Unigram))
input2Bigram.extend(nltk.bigrams(input2Unigram))
 
#Define Frequency Distributions
fdBiHam = FreqDist(filteredBiHam)
fdBiSpam = FreqDist(filteredBiSpam)
fbUniHam =  FreqDist(hamUnigrams)
fbUniSpam =  FreqDist(spamUnigrams)
uniqueCountHamUni = len(set(hamUnigrams))
uniqueCountSpamUni = len(set(spamUnigrams))
inputProHam = 0
inputProSpam = 0

#Probability calculation
def hamProbabilityCal(inputBigrams,tag):
    inputFreq = 1
    biFreq = 0
    for bigram in inputBigrams:
        biFreq = fdBiHam[bigram]
        firstWordFreq = 0 
        for Hbigrams in filteredBiHam:
            if Hbigrams[0] == bigram[0]:
                firstWordFreq += (fdBiHam[Hbigrams] + uniqueCountHamUni)
        wordFreq =  (biFreq +1) / firstWordFreq;      
        inputFreq *= wordFreq 
    inputProHam = inputFreq
    print(tag + "  HAM")
    print(inputFreq)
    return inputProHam

    
    
def spamProbabilityCal(inputBigrams,tag):
    inputFreq = 1
    biFreq = 0
    for bigram in inputBigrams:
        biFreq = fdBiSpam[bigram]
        firstWordFreq = 0 
        for Sbigrams in filteredBiHam:
            if Sbigrams[0] == bigram[0]:
                firstWordFreq += (fdBiSpam[Sbigrams] + uniqueCountSpamUni)
        wordFreq =  (biFreq +1) / firstWordFreq;      
        inputFreq *= wordFreq
    inputProSpam = inputFreq
    print(tag + "SPAM")
    print(inputFreq)
    return inputProSpam

   

def checkSpamOrHam(inputProHam,inputProSpam,inputSnt):
    if(inputProHam > inputProSpam):
        print(inputSnt +"----------> HAM ")
    elif(inputProHam < inputProSpam):
        print(inputSnt +"----------> SPAM ")
    elif(inputProHam == inputProSpam):
        print(inputSnt +"---------->Sorry, can not be categorized ")
    
inputProHam = hamProbabilityCal(input1Bigram,"Sentence 1 Probability ")
inputProSpam = spamProbabilityCal(input1Bigram,"Sentence 1 Probability ")
checkSpamOrHam(inputProHam,inputProSpam,inputS1)

inputProHam = hamProbabilityCal(input2Bigram,"Sentence 2 Probability ")
inputProSpam = spamProbabilityCal(input2Bigram,"Sentence 2 Probability ")
checkSpamOrHam(inputProHam,inputProSpam,inputS2)

