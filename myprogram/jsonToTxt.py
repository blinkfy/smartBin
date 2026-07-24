import os,json
list=os.listdir("./")
labelnames=['potato','rabbitcut','battery1','battery5','bottle','brick','can','china','stone','drug','drugbag','mooli','battery','bottle2','drugbox','papercup']
for name in list:
    if name[-4:]=='json':
        fp=open(name)
        data=json.load(fp)
        fp.close()
        ftxt=open(name[:-4]+".txt",'w')
        for data2 in data['shapes']:
            ftxt.write(str(labelnames.index(data2['label'])))
            for data3 in data2['points']:
                ftxt.write(f' {data3[0]/480} {data3[1]/480}')
            ftxt.write('\n')
        ftxt.close()
        