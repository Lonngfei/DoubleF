import csv
from glob import glob
from obspy import UTCDateTime



station_file = {}
w = open('Stations/stations.csv', 'w')
writer = csv.writer(w)
writer.writerow(['network', 'station', 'latitude', 'longitude', 'elevation'])



with open('Stations/station.dat') as s:
    for i in s:
        ss = i.strip('\n').split()
        lon, lat, net, sta, cha, alt = ss
        station_file[net+'_'+sta] = [lat, lon, cha, alt]
        writer.writerow([net, sta, lat, lon, alt])
w.close()

year0 = 2021
month0 = 10
day0 = 1
nday = 31

for i in range(nday):
    time = UTCDateTime(year0, month0, day0) + (i * 86400)
    year = time.year
    month = time.month
    day = time.day
    time0 = UTCDateTime(year, month, day)
    w = open(f'Picks/{year}{month:02d}{day:02d}.csv', 'w')
    writer = csv.writer(w)
    writer.writerow(["network", "station",  "latitude", "longitude", "elevation", "phasetype", "UTCTime", 'RelativeTime', 'Probability', 'Amplitude'])
    for file in glob(f'Picks/{year}{month:02d}{day:02d}/*.txt'):
        date = file.split('/')[-2]
        filename = file.split('/')[-1]
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])
        net, sta, type, _ = filename.split('.')
        datetime = UTCDateTime(year=year, month=month, day=day)
        with open(file) as f:
            for line in f:
                lines = line.strip('\n').split()
                sec_time, prob, amp = lines[:3]
                if sec_time == '(null)':
                    continue
                UTCTime = datetime + float(sec_time)
                RelativeTime = UTCTime - time0
                lat, lon, cha, alt = station_file[net+'_'+sta]
                amp = float(amp)
                result = [net, sta, lat, lon, alt, type, UTCTime, RelativeTime, prob, amp]
                writer.writerow(result)
    w.close()


