import csv
  
data = []
final_cloud_cover = []

with open('./train/train.csv', "r") as csvfile:
        lines = csv.reader(csvfile)
        writer = csv.writer(csvfile)
        for row in lines:
            
            cloud_cover = row[10]

            if cloud_cover != '-1' and cloud_cover != '-7999':
               final_cloud_cover.append(row) 

f = open('./train/train.csv', 'w',  newline='')

with f:

    writer = csv.writer(f)
    
    i = 0
    print(i)
    for row2 in final_cloud_cover:
        writer.writerow(final_cloud_cover[i])
        i = i + 1
f.close()

print('done')