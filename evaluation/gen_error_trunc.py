import json
import sys
import os

path = sys.argv[1]
all_files = os.listdir(path)

os.makedirs(f"{path}/rework/", exist_ok=True)
print(f"Saving to... >> {path}rework/")

for filename in all_files:
    if not filename.endswith("jsonl"):
        continue
    
    if filename not in ['narrativeqa.jsonl', 'hotpotqa.jsonl', 'qasper.jsonl', 'multi_news.jsonl', '2wikimqa.jsonl', 'gov_report.jsonl']:
        continue

    with open(f"{path}{filename}") as f :
        data = [json.loads(d) for d in f]
    
     
    try :
        with open(f"{path}rework/{filename}", 'r') as rf :
            current = [json.loads(d) for d in rf]
         
        if len(current) == len(data) :
            print(f"{filename} exists")
            continue
    except:
        current = list()
        print(f"No {filename}")
    
    with open(f"{path}rework/{filename}", 'a') as rf :
        print(f"Loaded {path}{filename}")
        current = len(current) 
        for line in data[current:] : 
            print("============================")
            print(line['pred'])
            print('============================')
            print("Label: ", line['label'], "\n")
            
            print(f"{current} / {len(data)}")
            ans = "n"
            while ans == 'n' :
                
                splitter = input("Split with >>> ")
                if splitter == "SKIP": 
                    temp = splitter 
                    break

                try :
                    temp = line['pred'].split(splitter)[0]
                except :
                    temp = line['pred']
                print("Result >>> ", temp)
                ans = input("OK? (y/n): ")
                


            if temp == "SKIP": 
                break

            line['pred'] = temp
            rf.write(json.dumps(line, ensure_ascii=False)+'\n')

            current += 1
            print("\n\n\n")
