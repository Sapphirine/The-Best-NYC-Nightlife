import re

with open("/Users/zhoupeiran/Documents/adb.csv","r") as reviewFile:
    f = reviewFile.readlines()


# Convert list to string
f = ''.join(f)

# Remove Special Characters, but keep space
f = re.sub(r'[^\w]', ' ', f)
# Remove mutiple spaces
f = re.sub(' +',' ', f)


# String save into a txt file
ff = open("/Users/zhoupeiran/Documents/adb2.csv","w")
ff.write(f)
ff.close()

# STOP words

with open("/Users/zhoupeiran/Documents/Reviews_Special_Characters.txt","r") as reviewFile:
    f = reviewFile.readlines()

f = ''.join(f)


pattern = re.compile("\\b(of|the|in|for|at|a|an|not|you|he|she|oh|and|to|it|is|I|with|that|was|my|this|then|have|been|me|has|by|or|are|no|we|be|can|t|all|our|want|this|what|on|just)\\W", re.I)
f = pattern.sub("", f)


ff = open("/Users/zhoupeiran/Documents/Reviews_Stopwords.txt","w")
ff.write(f)
ff.close()
