from sklearn.naive_bayes import GaussianNB
import numpy as np
import os
from six import BytesIO as StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


#convert pdf file to text function

class pdf_classify(object):
    
    def __init__(self,fname):
        self.fname = fname
        self.keyword_english = ["computer","economy","religion","social","technology"]
        self.result_dict = dict()
        #training data, it's a count for each keyword in keyword_english
        self.x = np.array([[0,0,0,41,0],[4,1,0,0,1],[1,1,0,0,5]])
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
        
    def find_all(self):
        self.convert()
        text_list = str(self.text.decode("utf8")).split(" ")
        for query in self.keyword_english:
            count = 0
            for i in text_list:
                if i==query:
                    count+=1
            self.result_dict[query] = count  
                              
    #classify the pdf using machine learning algorithm
    def classify(self):
        self.find_all()
        data = np.array([i for i in self.result_dict.values()][::-1]).reshape(1,-1)
                                
        gnb = GaussianNB()
        gnb.fit(self.x,self.Y)
        self.prediction = int(gnb.predict(data))
        
        if self.prediction == 1:
            print("%s is SOCIAL" %(self.fname))
        elif self.prediction == 2:
            print("%s is SCIENCE" %(self.fname))
        return self.prediction,data,self.result_dict
        
                   
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
    ebook = pdf_classify("something.pdf")
    ebook.classify()
