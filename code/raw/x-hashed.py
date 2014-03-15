import hashlib

m = hashlib.md5()
text = ""
for a in range(1,100):
    text = text
    m.update(text)
    m.hexdigest()
    text = m.hexdigest()
    print text

