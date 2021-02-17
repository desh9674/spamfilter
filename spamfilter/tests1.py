emails = """Subject: save your money buy getting this thing here  you have not tried cialls yet ?  than you cannot even imagine what it is like to be a real man in bed !  the thing is that a great errrectlon is provided for you exactly when you want .  ciaiis has a iot of advantaqes over viagra  - the effect iasts 36 hours !  - you are ready to start within just 10 minutes !  - you can mix it with aicohol ! we ship to any country !  get it riqht now ! . 
Subject: search engine position  be the very first listing in the top search engines immediately .  our company will now place any business with a qualified website  permanently at the top of the major search engines guaranteed never to move  ( ex : yahoo ! , msn , alta vista , etc . ) . this promotion includes unlimited  traffic and is not going to last long . if you are interested in being  guaranteed first position in the top search engines at a promotional fee ,  please contact us promptly to find out if you qualify via email at  searchl 1 @ telefonica . net . pe it ' s very important to include the url ( s ) if you  are interested in promoting ! ! ! this is not pay per click . examples will  be provided .  this promotion is only valid in the usa and canada .  sincerely ,  the search engine placement specialists  if you wish to be removed from this list , please respond to the following  email address and type the word " remove " in your subject line :  search 6 @ speedy . com . pe
"""
from collections import OrderedDict

def validate_input_text(intext):
    '''
    Validate the following details of input email text, provided for prediction.
    
    1. If the input email text contains more than one mail, they must be separated by atleast one blank line.
    
    2. Every input email must start with 'Subject:' pattern.
    
    Return False if any of the two validations fail.
    
    If all valiadtions pass, Return an Ordered Dicitionary, whose keys are first 30 characters of each
    input email and values being the complete email text.
    '''
    text = intext.strip()
    noOfEmails = text.count("Subject: ")
    noOfNewLines = text.count("\nSubject: ")
    #print(noOfNewLines)
    if noOfEmails>1 and noOfEmails-1==noOfNewLines: # or noOfEmails==noOfNewLines: ## first emails might be on newline or not, who knows, hence or conditions
        emails = text.split("\n")
        #print(emails)
        if False in [e.startswith("Subject: ") for e in emails]:
            print("False")
            return False
        else:
            newDict = OrderedDict()
            for e in emails:
                newDict[e[9:39]]= e
            #intro = OrderedDict([(e[9:39],e for e in emails)]) # chage to e[0:30]:e if required
            return newDict
    elif noOfEmails==1 and text.startswith("Subject: "):
        return OrderedDict([(text[9:39],text)])
    else:
        return False


print(validate_input_text(emails))
