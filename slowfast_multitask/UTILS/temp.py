import  csv

with open('train-for_user.csv', 'r', encoding='utf-8-sig') as f:
    dic = {}
    rdr = csv.reader(f)
    cnt=0
    for r in rdr:
        #print((r[-2]))
        d = r[-2]
        if d not in dic:
            dic[d]=0
        dic[d]+=1
        cnt+=1
    print(cnt)
    print(dic)
    '''
    cnt=0
    cnt2 = 0
    dic = {'Romance':0, 'Action':0, 'Horror':0, 'Crime':0,'Thriller':0,'Drama':0, 'Comedy':0, 'Adventure':0, 'Family':0}
    for r in rdr:
        for i in (r[9].split(',')):
            se.add(i.strip())
            if i.strip() in dic.keys():
                cnt+=1
                break
        cnt2+=1

        #for i in r[9].split(','):
            #if i in dic.keys():
                #cnt+=1
                #break
    print(cnt)
    print(cnt2)
    print(se)
    '''

