import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
model_path = 'dogs.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

print(input_shape)

# Load and preprocess the image
image_path = 'sheepdog.jpg'
image = Image.open(image_path).resize((input_shape[1], input_shape[2]))
image_array = np.asarray(image, dtype=np.float32)
image_array /= 255.0  # Normalize the pixel values between 0 and 1

# Prepare the input tensor
input_tensor = tf.convert_to_tensor(image_array)
input_tensor = np.expand_dims(input_tensor, axis=0)

# Set the input tensor to the model
interpreter.set_tensor(input_details[0]['index'], input_tensor)

# Run inference
interpreter.invoke()

# Get the output
output_tensor = interpreter.get_tensor(output_details[0]['index'])
predictions = output_tensor[0]

# Print or use the predictions as needed
print(predictions)

print ("Prediction: ", np.argmax(predictions))
index = np.argmax(predictions)

dogs = ["affenpinscher","afghan_hound","african_hunting_dog","airedale"
,"american_staffordshire_terrier","appenzeller","australian_terrier"
,"basenji","basset","beagle","bedlington_terrier","bernese_mountain_dog"
,"black-and-tan_coonhound","blenheim_spaniel","bloodhound","bluetick"
,"border_collie","border_terrier","borzoi","boston_bull"
,"bouvier_des_flandres","boxer","brabancon_griffon","briard"
,"brittany_spaniel","bull_mastiff","cairn","cardigan"
,"chesapeake_bay_retriever","chihuahua","chow","clumber","cocker_spaniel"
,"collie","curly-coated_retriever","dandie_dinmont","dhole","dingo"
,"doberman","english_foxhound","english_setter","english_springer"
,"entlebucher","eskimo_dog","flat-coated_retriever","french_bulldog"
,"german_shepherd","german_short-haired_pointer","giant_schnauzer"
,"golden_retriever","gordon_setter","great_dane","great_pyrenees"
,"greater_swiss_mountain_dog","groenendael","ibizan_hound","irish_setter"
,"irish_terrier","irish_water_spaniel","irish_wolfhound"
,"italian_greyhound","japanese_spaniel","keeshond","kelpie"
,"kerry_blue_terrier","komondor","kuvasz","labrador_retriever"
,"lakeland_terrier","leonberg","lhasa","malamute","malinois","maltese_dog"
,"mexican_hairless","miniature_pinscher","miniature_poodle"
,"miniature_schnauzer","newfoundland","norfolk_terrier"
,"norwegian_elkhound","norwich_terrier","old_english_sheepdog"
,"otterhound","papillon","pekinese","pembroke","pomeranian","pug"
,"redbone","rhodesian_ridgeback","rottweiler","saint_bernard","saluki"
,"samoyed","schipperke","scotch_terrier","scottish_deerhound"
,"sealyham_terrier","shetland_sheepdog","shih-tzu","siberian_husky"
,"silky_terrier","soft-coated_wheaten_terrier","staffordshire_bullterrier"
,"standard_poodle","standard_schnauzer","sussex_spaniel","tibetan_mastiff"
,"tibetan_terrier","toy_poodle","toy_terrier","vizsla","walker_hound"
,"weimaraner","welsh_springer_spaniel","west_highland_white_terrier"
,"whippet","wire-haired_fox_terrier","yorkshire_terrier"]

print (dogs[index])