import subprocess
import glob

wanted_tags = ["Volcano", "Sunrise Sunset", "ISS Structure", "Stars", "Night", "Aurora", "Movie", "Day", "Moon", "Inside ISS", "Dock Undock", "Cupola"]
filenames = glob.glob("../../BUSampleDataSet/*.jpg")

csv_str = ""
max_tags = 0
for filename in filenames:
    try: # use ExifTool from http://www.sno.phy.queensu.ca/~phil/exiftool/
        output = subprocess.check_output(['/usr/local/bin/exiftool', "-Subject", filename])
        all_tags = output.decode(encoding = 'utf_8')[34:].rstrip()
    except Exception:
        print('Error getting Exif data')
    if all_tags:
        tags = ""
        all_tags = all_tags.split(", ")
        count = 0
        for tag in all_tags:
            if tag in wanted_tags:
                tags += tag + ","
                count += 1
        if count > max_tags:
            max_tags = count
        print(tags)
        #windows
        # filename = filename[filename.rindex("\\")+1:]

        #mac
        filename = filename[filename.rindex("/")+1:]
        csv_str += filename + "," + tags[:-1] + "\n"
print(max_tags)
tag_columns = ""
for i in range(max_tags):
    tag_columns += "tag_" + str(i) + ","
csv_str = "id,"+ tag_columns[:-1] + "\n" + csv_str
with open('extracted_tags.csv', 'w') as f:
    f.write(csv_str)
    f.close()

