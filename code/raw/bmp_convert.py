from pil import Image
photo = raw_input("image-in: ")
out_photo = raw_input("image-out: ")
im = Image.open(photo)
im.save(out_photo, "bmp")