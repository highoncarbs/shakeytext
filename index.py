import csv
import nltk
import itertools

vocab_size = 8000
unknown_token = "UNKNOWN_TOKEN"
start_token = "SENTENCE_START"
end_token = "SENTENCE_END"
# Read the CSV File will_play_text.CSV;
print "Reading William Fucktards novels....  \n"

with open("will_play_text.csv", "rb") as f:
    reader = csv.reader(f)
    sentence = list(reader)
    for i in range(800):
        sentence[:800][i][5]
    #for i in xrange(900):
     #   sen = sentence[0:100][i][5]
 
