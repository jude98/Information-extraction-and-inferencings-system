# Information-extraction-and-inferencings-system
Drug information extraction from online malayalam newspaper and developed NER using Spcay



 ## Prerequisites

1. python3
   ```
   sudo apt install python3
   ```
2. pip3
    
    ```
    sudo apt install python3-pip
    ```
## Setting Up Your Environment
Install the following python packages `spacy`, `scrapy`
```
sudo pip3 install spacy
sudo pip3 install scrapy
```

## Running
### Scrapy 
```bash
cd ./newscrawl/newscrawl/spider/
scrapy crawl asianet -o news.csv

```
### Training Model
```bash
cd ./Information-extraction-and-inferencings-system
python3 train_ner.py
```

### Testing Model
```bash
cd ./Information-extraction-and-inferencings-system
python3 test_ner.py >> out.txt
```