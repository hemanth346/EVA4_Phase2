import os
import io
import zipfile
from PIL import Image
from pathlib import Path


name_dict = {'Flying Birds.zip':'bird', 'Small QuadCopters.zip': 's-quadcop', 'Winged Drones.zip':'drone', 'Large QuadCopters.zip':'l-quadcop'}
data_dir = '.'
dest_dir='processed'

if not os.path.isdir(dest_dir):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

for fname in os.listdir(data_dir):
    abs_fname = os.path.join(data_dir, fname)
    if fname in name_dict and zipfile.is_zipfile(abs_fname):
        # where the processed images has to be stored
        preprocessed_zip = zipfile.ZipFile(os.path.join(dest_dir, name_dict[fname]+'.zip'), mode='a', compression=zipfile.ZIP_STORED)
        # where the images are read from
        original_zip = zipfile.ZipFile(abs_fname) 
        # # debug logger
        # print('Writing to ',os.path.join(dest_dir, name_dict[fname]+'.zip'))
        for idx, zfile in enumerate(original_zip.infolist()):
            imgdata = original_zip.read(zfile.filename)
            try:
                img = Image.open(io.BytesIO(imgdata)).convert("RGB")
                if img:
                    # resize
                    img.thumbnail((500, 500)) # max width or height by handling aspect ratio
                    img.save('temp.jpg')
                    preprocessed_zip.write('temp.jpg', '{cls}_{num}.jpg'.format(cls=name_dict[fname], num=str(idx+1)))
                    # # debug logger
                    # print('{cls}_{num}.jpg'.format(cls=name_dict[fname], num=str(idx+1)))
            except:
                pass
        preprocessed_zip.close()
        original_zip.close()
        os.remove('temp.jpg')
