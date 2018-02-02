import face_recognition
import base64

# pip3 face_recognition
'''
php传入进来base64编码文件后，python 将文件读为 .jpg 图片 然后再进行比对
f = open('../testfile.txt')
txt = f.read()
img = base64.b64decode(txt)
cnt = open('../images/b64.jpg', 'wb')
cnt.write(img)
cnt.close()
quit()
'''

# Load the jpg files into numpy arrays
# 数据库中参与对比的图片(被比较的图片)
biden_image = face_recognition.load_image_file("../images/yzs2.jpg")
obama_image = face_recognition.load_image_file("../images/zj.jpg")

# 前去对比的图片
unknown_image = face_recognition.load_image_file("../images/yzs.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    biden_face_encoding,
    obama_face_encoding
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

print("Is the unknown face a picture of wo? {}".format(results[0]))
print("Is the unknown face a picture of me? {}".format(results[1]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))