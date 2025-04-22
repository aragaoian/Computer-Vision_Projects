import psycopg2
import cv2
import numpy as np


# cursor.execute("""
#     CREATE TABLE IF NOT EXISTS images (
#         id SERIAL PRIMARY KEY,
#         name TEXT,
#         data BYTEA
#     )
# """)
# conn.commit()

# def convert_to_binary(filename):
#     with open(filename, 'rb') as file:
#         binary_data = file.read()
#     return binary_data

# image_name = "perceptron.png"
# image_data = convert_to_binary('/home/ian/Univali/Machine Learning/ComputerVision_Projects/FaceRecognition/faces/perceptron.png')

# cursor.execute("INSERT INTO images (name, data) VALUES (%s, %s)", (image_name, image_data))
# conn.commit()

# cursor.close()
# conn.close()

cursor.execute("SELECT data FROM images WHERE name = 'perceptron.png' ")
image_data = cursor.fetchone()[0]  # Extract the binary data


nparr = np.frombuffer(image_data, np.uint8)
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode the image

# output_image_path = 'retrieved_image.jpg'
# cv2.imwrite(output_image_path, image)


cv2.imshow("Retrieved Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


cursor.close()
conn.close()

