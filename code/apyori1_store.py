# -*- coding: utf-8 -*-
from apyori import apriori
import pandas as pd
data = [['豆奶','萵苣'], ['萵苣','尿布','葡萄酒','甜菜'], ['豆奶','尿布','葡萄酒','橙汁'],
['萵苣','豆奶','尿布','葡萄酒'], ['萵苣','豆奶','尿布','橙汁']]

min_supp = 0.5  #设定最小支持度
min_conf = 0.5   #设定最小置信度
min_lift = 1.01    #设定最小提升度
min_length=2

rules = list(apriori(transactions=data, min_support=min_supp,
           min_confidence=min_conf, min_lift=min_lift, min_length=min_length))

print(len(rules))

supports=[]
confidences=[]
lifts=[]
bases=[]
adds=[]

for r in rules:
    for x in r.ordered_statistics:
        bases.append(list(x.items_base))
        adds.append(list(x.items_add))
        supports.append(r.support)
        confidences.append(x.confidence)
        lifts.append(x.lift)


#將結果存為dataframe
result = pd.DataFrame({
    'base':bases,
    'add':adds,
    'support':supports,
    'confidence':confidences,
    'lift':lifts
})

print(result)

