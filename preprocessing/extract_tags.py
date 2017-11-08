import subprocess
import glob

wanted_tags = ["Volcano", "Sunrise Sunset", "ISS Structure", "Stars", "Night", "Aurora", "Movie", "Day", "Moon", "Inside ISS", "Dock Undock", "Cupola"]
filenames = glob.glob("./BUSampleDataSet/*.jpg")

csv_str = ""
for filename in filenames:
    try: # use ExifTool from http://www.sno.phy.queensu.ca/~phil/exiftool/
        output = subprocess.check_output(['/usr/local/bin/exiftool', "-Subject", filename])
        all_tags = output.decode(encoding = 'utf_8')[34:].rstrip()
    except Exception:
        print('Error getting Exif data')
    if all_tags:
        tags = ""
        all_tags = all_tags.split(", ")
        for tag in all_tags:
            if tag in wanted_tags:
                tags += tag + ","
        print(tags)
        filename = filename[filename.rindex("/")+1:]
        csv_str += filename + "," + tags[:-1] + "\n"
csv_str = "id\n" + csv_str
with open('extracted_tags.csv', 'w') as f:
    f.write(csv_str)
    f.close()

