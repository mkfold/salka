import os
import pandas as pd

def main():

    #output file
    # outputPath = os.path.join(os.getcwd(), "convert_data")
    #
    # if not os.path.exists(outputPath):
    #     os.makedirs(outputPath)

    f = open("convert_data1.csv", 'w')

    #input file
    inputPath = os.path.join(os.getcwd(), "byte_logstash-2018.04.30.csv")
    content = open(inputPath).read()

    count = 0
    for line in content:

     line = line.strip(" ")
     if line == "\n":
         f.write(str(count) + '\n')
         count = 0
     else:
         line = ord(line)
         count = count + 1
         f.write(str(line) + ' ')





    #tokens = nltk.word_tokenize(content)
    #print(tokens)



if __name__ == "__main__":

    main()
