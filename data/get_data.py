import os

for file in os.listdir('content/BCCD/train'):
  if file == 'annos' or file == 'images': continue
  if file.split(sep='.')[3] == 'jpg':
    os.rename("content/BCCD/train/"+file, "content/BCCD/train/images/"+file)
  else: os.rename("content/BCCD/train/"+file, "content/BCCD/train/annos/"+file)


  for file in os.listdir('content/BCCD/valid'):
      if file == 'annos' or file == 'images': continue
      if file.split(sep='.')[3] == 'jpg':
          os.rename("content/BCCD/valid/" + file, "content/BCCD/valid/images/" + file)
      else:
          os.rename("content/BCCD/valid/" + file, "content/BCCD/valid/annos/" + file)

  for file in os.listdir('content/BCCD/test'):
      if file == 'annos' or file == 'images': continue
      if file.split(sep='.')[3] == 'jpg':
          os.rename("content/BCCD/test/" + file, "content/BCCD/test/images/" + file)
      else:
          os.rename("content/BCCD/test/" + file, "content/BCCD/test/annos/" + file)