import  csv

with open('train-for_user.csv', 'r', encoding='utf-8-sig') as f:
    se=set()
    rdr = csv.reader(f)
    for r in rdr:
        se.add(r[2])
    print(se)
