import cv2
import os

# Set the current working directory to the directory containing the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier(r'C:\Users\Harshit Singh\Downloads\Compressed\OpenCV-Face-Recognition-master\OpenCV-Face-Recognition-master\FaceDetection\haarcascade_frontalface_default.xml')

while True:
    # Get user info
    first_name = input("\nEnter first name: ")
    last_name = input("Enter last name: ")
    middle_name = input("Enter middle name (optional, leave blank if none): ")

    # Combine first, middle, and last names (if middle name is provided)
    if middle_name:
        folder_name = f"{first_name}_{middle_name}_{last_name}"
    else:
        folder_name = f"{first_name}_{last_name}"

    # Create folder with the user's name
    folder_path = os.path.join(r'C:\Users\Harshit Singh\Downloads\DC\lfw\lfw', folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Ask user to confirm folder name
    print(f"\nFolder will be created at:\n{folder_path}")
    confirm = input("Press 'y' to confirm or any other key to retry: ")
    if confirm.lower() == "y":
        break

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while True:

    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the user's folder
        image_path = os.path.join(folder_path, f"{folder_name}_{count}.jpg")
        cv2.imwrite(image_path, gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 5: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()



