import csv
import nltk
import itertools

vocab_size = 8000
unknown_token = "UNKNOWN_TOKEN"
start_token = "SENTENCE_START"
end_token = "SENTENCE_END"
# Read the txt File will_new.txt;
print "Reading William Fucktards novels....  \n"

with open("will_new.txt", "rb") as f:
    reader = f.read()
    words = nltk.word_tokenize(reader)
    
    print words
    #sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    #sentences = ["%s %s %s" %(start_token , x , end_token) for x in sentences]
