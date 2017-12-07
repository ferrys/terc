import os
import subprocess
import csv


def insert_tags(predictions_file, image_dir):
	tags = ['Volcano', 'Sunrise Sunset', 'ISS Structure', 'Stars', 'Night', 'Aurora', 'Movie', 'Day', 'Moon', 'Inside ISS', 'Dock Undock', 'Cupola']

	with open(predictions_file, 'rt') as f:
		reader = csv.reader(f, delimiter = ',')
		next(reader, None)
		for row in reader:
			file_name = row[0]
			image = image_dir+'/'+file_name

			predicted_tags = row[1:]
			for i, tag in enumerate(predicted_tags):
				if tag == '1':
					command = '-Keywords+=' + tags[i]
					subprocess.call(['exiftool', command, image], shell=True)
	f.close()

	tagged_images = os.listdir(image_dir)
	for image in tagged_images:
	    if image.endswith(".jpg_original"):
	        os.remove(os.path.join(image_dir, image))