file_name = "testB.txt"
f = open(file_name, 'w')
for i in range(1000):
    img_path = str(i+4000+1).zfill(5) + '.png'
    f.write(img_path + '\n')
f.close()

