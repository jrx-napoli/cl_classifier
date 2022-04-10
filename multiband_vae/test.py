txt = ""
x = 0

for i in range(1, 10000):
    
    txt += str(i)
    if len(txt) == 2040: 
        x = i
        break

print(x)