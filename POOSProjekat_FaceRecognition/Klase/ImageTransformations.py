from PIL import Image
image_file = Image.open(r"C:\Users\mali_cox\Pictures\test.jpg") # open colour image
image_file = image_file.convert('1') # convert image to black and white
image_file.save('result.png')

#Ovo je test metoda samo da provjerim da li radi ovo