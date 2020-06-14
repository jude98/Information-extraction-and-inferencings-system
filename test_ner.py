import spacy
from pathlib import Path
import csv


def check_similarity(text,values):
    #Check the similarities for repeating Entities
    f=0
    for each in values.keys():
        s_index = 0
        for i in text:
            if i in each:
                s_index = s_index + 1
        if s_index >= max(len(each),len(text))/3:
            f = 1
            break
    if f == 0:
        return False
    else:
        return True





def main():

        #Trained model is saved in directory train_ner

        output_dir="train_ner"
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)


        #Sample news given as input

        # new_text = "താമരശ്ശേരി: പ്രായപൂർത്തിയാകാത്ത പെൺകുട്ടിയെ , നൽകി പീഡിപ്പിച്ചു. പത്തൊമ്പതുകാരൻ അറസ്റ്റിൽ. കൊടിയത്തൂർ സ്വദേശി സിടി അഷ്റഫിനെയാണ് പോലീസ് അറസ്റ്റ് ചെയ്തത്. താമരശ്ശേരി കോടതിയിൽ ഹാജരാക്കിയ അഷ്റഫിനെ 14 ദിവസത്തേക്ക് റിമാന്റ് ചെയ്യും. മുക്കം നഗരസഭാ പരിധിയിലെ സര്‍ക്കാര്‍ സ്കൂളിലെ വിദ്യാർത്ഥിയാണ് പീഡനത്തിനിരയാക്കിയത്.,സര്‍ക്കാര്‍ സ്കൂളിൽ ശുചിമുറിയില്‍ 16കാരി സിഗരറ്റ് വലിക്കുന്നത് കണ്ടത് സഹപാഠികൾ അധ്യാപകരെ അറിയിച്ചു. അധ്യാപകര്‍ ചോദ്യം ചെയ്തപ്പോള്‍ സിഗരറ്റ് നൽകിയത് അഷ്റഫാണെന്ന് പെൺകുട്ടി അറിയിച്ചു. ഇതോടെ, സ്കൂൾ അധികൃതര്‍ മുക്കം പോലീസില്‍ പരാതി നല്‍കി.,കഞ്ചാവടക്കമുള്ള ലഹരി മരുന്ന് നല്‍കി അഷ്റഫ് പലതവണ പീഡിപിക്കാന്‍ ശ്രമിച്ചിട്ടുണ്ടെന്നാണ് പെണ്കുട്ടി പോലീസിന് നൽകിയ മോഴിയിൽ വ്യക്തമാക്കുന്നത്. പ്രദേശത്തെ സ്കൂളുകളും കോളേജുകളും കേന്ദ്രീകരിച്ച് ലഹരി വിൽപ്പന നടത്തുന്ന സംഘത്തിലെ പ്രധാന കണ്ണിയാണ് അഷ്റഫ് എന്നും പോലീസ് വ്യക്തമാക്കുന്നു. അഷ്റഫിന് പിന്നിൽ വലിയ സംഘമുണ്ടെന്നും ഇവർക്കായി തിരച്ചിൽ തുരുകയാണെന്നും പോലീസ് വ്യക്തമാക്കി."

        #Reading news_contents from crawled newscrawl
        #Path to the crawled news
        file_oneindia = Path("../newscrawl/newscrawl/spiders/news.csv")
        all_news = []


        #News of Oneindia/Asianet stored in csv file
        with open(file_oneindia,'r') as one_india:
        	r_oneindia = csv.DictReader(one_india)
        #    r_asianet = csv.DictReader(asianet)
        	for row in r_oneindia:
        		all_news.append(row)

        for new_text in all_news:
            # print(new_text['content'])
            #The news is given to the variable new_text
            doc = nlp2(new_text['content'])

            #Entities
            # for ent in doc.ents:
            #     print(ent.label_, ent.text)
            drugs = {}
            locations=[]
            person={}
            date=[]
            time=[]
            pc=1

            #Findind the location of crime
            for ent in doc.ents:
                if ent.label_ == 'Location':
                    print("The location of the news : ",ent.text)
                    break
            i = 1
            while i < len(doc.ents):
                #Finding the drugs details for Quantity/Money -> Drugs

                if doc.ents[i].label_ == 'Quantity' or doc.ents[i].label_ == 'Money':
                    if i+1 < len(doc.ents):
                        if doc.ents[i+1].label_ == 'drug':
                            drugs[doc.ents[i+1].text] = doc.ents[i].text + '(' + doc.ents[i].label_ + ')'
                            i = i+2
                        else:
                            i = i+1
                    else:
                        i = i+1

                #Findind Drugs details of which quantity not mentioned(QNM)
                elif doc.ents[i].label_ == 'drug':
                    if(check_similarity(doc.ents[i].text,drugs) == False):
                        drugs[doc.ents[i].text] = 'QNM'
                    i = i+1

                #Findind rest of the locations mentioned in the news
                elif doc.ents[i].label_ == 'Location':
                    locations.append(doc.ents[i].text)
                    i = i+1

                #Findind person details for Age -> person or age alone
                elif doc.ents[i].label_ == 'Age':
                    if doc.ents[i+1].label_ == 'Person':
                        person[doc.ents[i+1].text] = doc.ents[i].text
                        i = i+2
                    else:
                        person['Unknown_person'+str(pc)] = doc.ents[i].text
                        pc = pc+1
                        i =i+1

                #Findind person details for person -> age
                elif doc.ents[i].label_ == 'Person':
                    if i+1 < len(doc.ents):
                        if doc.ents[i+1].label_ == 'Age':
                            person[doc.ents[i].text] = doc.ents[i+1].text
                            i = i+2
                        else:
                            if check_similarity(doc.ents[i].text,person) == False:
                                person[doc.ents[i].text] = 'ANM'
                            i = i+1
                    else:
                        i = i+1

                #Date and time details
                elif doc.ents[i].label_ == 'Date':
                    date.append(doc.ents[i].text)
                    i = i+1
                elif doc.ents[i].label_ == 'Time':
                    time.append(doc.ents[i].text)
                    i = i+1
                else:
                    i = i+1

            print("Drug details : ",drugs,"\nAssociated persons : ",person,"\nDate : ",date,"\ntime : ",time,"\nOther locations : ",locations,"\n")



    #for ent in doc.ents:
    #    print(ent.label_, ent.text)


if __name__ == '__main__':
    main()
