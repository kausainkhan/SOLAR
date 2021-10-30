import csv
  
data = []
with open('data/train/train/train.csv', "r") as csvfile:
        lines = csv.reader(csvfile)
        writer = csv.writer(csvfile)
        for row in lines:
            
            mmdd = row[0].split('/')
            hhmm = row[1].split(':')

            if len(mmdd[0]) == 1:
                mmdd.insert(0, '0')
                if len(mmdd[2]) == 1:
                    mmdd.insert(2, '0')

            elif len(mmdd) > 1:
                if len((mmdd[1])) == 1:
                    mmdd.insert(1, '0')

            mmdd = ''.join(mmdd)

            hhmm = ''.join(hhmm)
            data.append([mmdd, hhmm, row[2], row[3], row[4], row[5],  row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16]])

f = open('data/train/train/train.csv', 'w', newline='')

with f:

    writer = csv.writer(f)
    
    i = 0
    print(i)
    for row in data:
        if row[0] == 'DATE (MMDD)':
            row.insert(0, 'Time [Mins]')
            writer.writerow(row)
            row.pop(0)
        else:
            row.insert(0, i - 1)
            writer.writerow(row)
        i = i + 1

print('done')