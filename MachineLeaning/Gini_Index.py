def gini_index_single(a,b):
    single_gini=1-(a/(a+b))**2-(b/(a+b))**2
    return round(single_gini,5)

def gini_index(a,b,c,d):
    left=gini_index_single(a,b)
    right=gini_index_single(c,d)
    gini_index=left*((a+b)/(a+b+c+d))+right*((c+d)/(a+b+c+d))
    return round(gini_index,3)

print(gini_index(105,39,34,125))
print(gini_index(37,127,100,33))
print(gini_index(92,31,45,129))