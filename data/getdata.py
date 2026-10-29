import urllib.request
import os

out = 'student-mat.csv'
urls = [
    'https://raw.githubusercontent.com/DenisOuma/Student-Performance-Data-Set/master/student-mat.csv',
    'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/student-mat.csv',
]
for u in urls:
    try:
        urllib.request.urlretrieve(u, out)
        print('ok')
        break
    except Exception as e:
        print(e)
        continue
else:
    # fallback synthetic if nothing works
    import csv, random
    random.seed(123)
    fields = ['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3']
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i in range(395):
            row = ['GP','F' if i%2==0 else 'M', 15+i%5,'U','GT3','T', random.randint(1,4), random.randint(1,4),'at_home','teacher','course','mother',1,2,0,'no','yes','no','no','yes','yes','yes','no',random.randint(1,5),random.randint(1,5),random.randint(1,5),1,1,random.randint(1,5),random.randint(0,10), random.randint(0,20), random.randint(0,20), random.randint(0,20)]
            w.writerow(row)
    print('synthetic')
