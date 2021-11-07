import csv
  


for k in range(300):
    data = []
    final_cloud_cover = []

    with open(f'./pred/{k + 1}/weather_data.csv', "r") as csvfile:
            lines = csv.reader(csvfile)
            writer = csv.writer(csvfile)
            for row in lines:
                
                cloud_cover = row[8]

                if cloud_cover != '-1' and cloud_cover != '-7999':
                    final_cloud_cover.append(row) 

    f = open(f'./pred/{k + 1}/weather_data.csv', 'w',  newline='')

    with f:

        writer = csv.writer(f)
        
        i = 0
        print(i)
        for row2 in final_cloud_cover:
            writer.writerow(final_cloud_cover[i])
            i = i + 1
    f.close()

print('done')