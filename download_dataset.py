import gdown
import rarfile

url = "https://drive.google.com/file/d/18UlvYDL6UGZ2rs0yaDsSzoUlw8KI5ABY/view"
output = "dataset.rar"
gdown.download(url, output, quiet=False, fuzzy=True)

with rarfile.RarFile('/content/GAN-i-do-makeup/dataset.rar', 'r') as rf:
    for member in rf.infolist():
        if member.filename.startswith('all/images/'):
            rf.extract(member, '/content/GAN-i-do-makeup/dataset/')
