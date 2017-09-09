from sklearn.naive_bayes import GaussianNB
import numpy as np
import os
from six import BytesIO as StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from threading import Thread


#convert pdf file to text function
def convert(fname):
    pagenums = set()
    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)
    
    infile = open(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close
    return text 

def find_text(query,text,result_dict):
    text_list = str(text.decode("utf8")).split(" ")
    count = 0
    for i in text_list:
        if i==query:
            count+=1
    result_dict[query] = count        
    
#classify the pdf using machine learning algorithm
def classification(class_dict):    
    data = np.array([i for i in class_dict.values()][::-1]).reshape(1,-1)
    x = np.array([[0,0,0,41,0],[4,1,0,0,1],[1,1,0,0,5]])
    Y = np.array([1,2,2])                        
    gnb = GaussianNB()
    gnb.fit(x,Y)
    return int(gnb.predict(data))

def classify(classification):
    if classification == 1:
        print("SOCIAL")
    elif classification == 2:
        print("SCIENCE")
        
def get_list(directory):
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
    pdf_name = input("input pdf file name: ")
    text = convert(pdf_name)
    keyword_english = ["economy","social","technology","religion","computer"]
    
    class_dict = dict()
    
    threads = []
    for i in keyword_english:
        find_text(i,text,class_dict)
        t = Thread(target=find_text,args=(i,text,class_dict))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()    
                
    classify(classification(class_dict))
