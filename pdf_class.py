from sklearn.naive_bayes import GaussianNB
import numpy as np
import os
from six import BytesIO as StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from google.cloud import translate
from textblob import TextBlob
#convert pdf file to text function

class pdf_classify(object):
    
    def __init__(self,fname):
        self.fname = fname
        self.keyword_english = ["computer","economy","religion","social","technology","nuclear","computation","calculations"]
        self.result_list = []
        #training data, it's a count for each keyword in keyword_english
        self.x = np.array([[0,0,0,41,0,0,0,0],[4,1,0,0,1,1,1,1],[1,1,0,0,5,2,1,1]])
        self.Y = np.array([1,2,2])
    
    #convert pdf to text 
    def convert(self):
        pagenums = set()
        output = StringIO()
        manager = PDFResourceManager()
        converter = TextConverter(manager, output, laparams=LAParams())
        interpreter = PDFPageInterpreter(manager, converter)        
        infile = open(self.fname, 'rb')
        for page in PDFPage.get_pages(infile, pagenums):
            interpreter.process_page(page)
        infile.close()
        converter.close()
        self.text = output.getvalue()
        output.close
        return self.text
    
    def detect_language(self):
        txt = str(self.text.decode("utf8"))[:100]
        b = TextBlob(txt)
        self.lang = b.detect_language()
        return b.detect_language()

   
    def find_all(self):
        self.convert()
        self.detect_language()
        
        text_list = str(self.text.decode("utf8")).split(" ")
        #keywords_txt = " ".join(self.keyword_english)
        
        list_ = []
        if self.lang != "en":
            
            for text in self.keyword_english:
                blob = TextBlob(text)
                try:
                    list_.append(str(blob.translate(to=self.lang)))
                except:
                    list_.append(text)
        else:
            list_ = self.keyword_english
            
        for query in list_:
            count = 0
            for i in text_list:
                if i==query:
                    count+=1        
            self.result_list.append((query,count))  
        return list_
                  
    #classify the pdf using machine learning algorithm
    def classify(self):
        self.find_all()
        data = np.array([self.result_list[i][1] \
                         for i in range(len(self.result_list))]).reshape(1,-1) 
                               
        gnb = GaussianNB()
        gnb.fit(self.x,self.Y)
        self.prediction = int(gnb.predict(data))
        
        if self.prediction == 1:
            print("%s is SOCIAL SCIENCE" %(self.fname))
        elif self.prediction == 2:
            print("%s is NATURAL SCIENCE" %(self.fname))
        return self.prediction,data,self.result_list
        
                   
def get_list(directory=os.getcwd()):
    pdflist=[]
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"): 
            #print(os.path.join(directory, filename))
            pdflist.append(filename)
            continue
        else:
            continue
    return pdflist 

if __name__ == "__main__":
    ebook = pdf_classify("manifesto.pdf")
    ebook.convert()
    c = ebook.classify()
