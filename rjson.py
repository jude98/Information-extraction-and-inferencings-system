import json

with open('Data.json') as f:
    data = json.loads("[" + 
        f.read().replace("}\n{", "},\n{") + 
    "]")

new_data=[]
for each_data in data:
	d={}
	d['labels']=each_data['labels']
	new_data.append((each_data['text'],d))
print(new_data)
