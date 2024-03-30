import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io
import tensorflow_hub as hub
import base64
import pandas as pd

label_dict = {0: 'affenpinscher',
    1: 'afghan_hound',
    2: 'african_hunting_dog',
    3: 'airedale',
    4: 'american_staffordshire_terrier',
    5: 'appenzeller',
    6: 'australian_terrier',
    7: 'basenji',
    8: 'basset',
    9: 'beagle',
    10: 'bedlington_terrier',
    11: 'bernese_mountain_dog',
    12: 'black-and-tan_coonhound',
    13: 'blenheim_spaniel',
    14: 'bloodhound',
    15: 'bluetick',
    16: 'border_collie',
    17: 'border_terrier',
    18: 'borzoi',
    19: 'boston_bull',
    20: 'bouvier_des_flandres',
    21: 'boxer',
    22: 'brabancon_griffon',
    23: 'briard',
    24: 'brittany_spaniel',
    25: 'bull_mastiff',
    26: 'cairn',
    27: 'cardigan',
    28: 'chesapeake_bay_retriever',
    29: 'chihuahua',
    30: 'chow',
    31: 'clumber',
    32: 'cocker_spaniel',
    33: 'collie',
    34: 'curly-coated_retriever',
    35: 'dandie_dinmont',
    36: 'dhole',
    37: 'dingo',
    38: 'doberman',
    39: 'english_foxhound',
    40: 'english_setter',
    41: 'english_springer',
    42: 'entlebucher',
    43: 'eskimo_dog',
    44: 'flat-coated_retriever',
    45: 'french_bulldog',
    46: 'german_shepherd',
    47: 'german_short-haired_pointer',
    48: 'giant_schnauzer',
    49: 'golden_retriever',
    50: 'gordon_setter',
    51: 'great_dane',
    52: 'great_pyrenees',
    53: 'greater_swiss_mountain_dog',
    54: 'groenendael',
    55: 'ibizan_hound',
    56: 'irish_setter',
    57: 'irish_terrier',
    58: 'irish_water_spaniel',
    59: 'irish_wolfhound',
    60: 'italian_greyhound',
    61: 'japanese_spaniel',
    62: 'keeshond',
    63: 'kelpie',
    64: 'kerry_blue_terrier',
    65: 'komondor',
    66: 'kuvasz',
    67: 'labrador_retriever',
    68: 'lakeland_terrier',
    69: 'leonberg',
    70: 'lhasa',
    71: 'malamute',
    72: 'malinois',
    73: 'maltese_dog',
    74: 'mexican_hairless',
    75: 'miniature_pinscher',
    76: 'miniature_poodle',
    77: 'miniature_schnauzer',
    78: 'newfoundland',
    79: 'norfolk_terrier',
    80: 'norwegian_elkhound',
    81: 'norwich_terrier',
    82: 'old_english_sheepdog',
    83: 'otterhound',
    84: 'papillon',
    85: 'pekinese',
    86: 'pembroke',
    87: 'pomeranian',
    88: 'pug',
    89: 'redbone',
    90: 'rhodesian_ridgeback',
    91: 'rottweiler',
    92: 'saint_bernard',
    93: 'saluki',
    94: 'samoyed',
    95: 'schipperke',
    96: 'scotch_terrier',
    97: 'scottish_deerhound',
    98: 'sealyham_terrier',
    99: 'shetland_sheepdog',
    100: 'shih-tzu',
    101: 'siberian_husky',
    102: 'silky_terrier',
    103: 'soft-coated_wheaten_terrier',
    104: 'staffordshire_bullterrier',
    105: 'standard_poodle',
    106: 'standard_schnauzer',
    107: 'sussex_spaniel',
    108: 'tibetan_mastiff',
    109: 'tibetan_terrier',
    110: 'toy_poodle',
    111: 'toy_terrier',
    112: 'vizsla',
    113: 'walker_hound',
    114: 'weimaraner',
    115: 'welsh_springer_spaniel',
    116: 'west_highland_white_terrier',
    117: 'whippet',
    118: 'wire-haired_fox_terrier',
    119: 'yorkshire_terrier'
}


@st.cache_resource
def load_model():
    # Explicitly use custom_object_scope for KerasLayer from TensorFlow Hub
    with tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
        model = tf.keras.models.load_model('20230625-04441687668282-all-images-Adam.h5')
    return model
    
model = load_model()

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)[0]  # Get predictions for the image
    
    # Get the indices of the top 5 predicted classes
    top_5_indices = np.argsort(prediction)[::-1][:5]
    
    # Create a table to display the predictions
    table_data = []
    for idx in top_5_indices:
        breed_label = label_dict[idx]
        probability = prediction[idx]
        table_data.append([breed_label, f"{probability:.2%}"])
    
    # Convert the table data to a DataFrame
    table_df = pd.DataFrame(table_data, columns=["Breed", "Probability"])
    
    return table_df


def run():
    img1 = Image.open('logo.jpg')
    img1 = img1.resize((700, 350))
    st.image(img1, use_column_width=False)

    st.markdown(
        """
        <h1 style="text-align: center;">DOG VISION</h1>
        <h4 style="text-align: center; color: #d73b5c;">The trained data consists of a collection of 10,000+ labeled images of 120 different dog breeds.</h4>
        """,
        unsafe_allow_html=True
    )

    st.markdown('---')

    st.markdown(
        """
        <h3 style="text-align: center;">Upload an Image</h3>
        <p style="text-align: center;">Please upload an image of a dog to analyze its breed.</p>
        """,
        unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.text("Please upload an image file!")
    else:
        img = Image.open(io.BytesIO(uploaded_file.read()))

        # Center-align the image
        img_str = img_to_base64(img)
        st.markdown(
            f'<div style="text-align: center;"><img src="data:image/png;base64,{img_str}" alt="Uploaded Image" width="400px"></div>',
            unsafe_allow_html=True
        )
        st.markdown('---')

        st.success('Image uploaded successfully!')

        table_data = import_and_predict(img, model)
        
        # Display the table of predicted breeds and probabilities
        st.table(table_data)
        st.markdown('---')

        breed_label = table_data["Breed"].iloc[0]
        
        # Display the top probability output breed in the specified format
        st.markdown("<h2 style='text-align: center;'><span style='color: orange;'>Predicted Breed :  </span><span style='color: green;'>{}</span></h2>".format(breed_label), unsafe_allow_html=True)

        st.markdown('---')
        
        # Provide a clickable link to open Google search results
        search_query = f"{breed_label} dog images"
        search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
        link_html = f'<div style="text-align: center;"><a href="{search_url}" target="_blank" style="display: inline-block; text-align: center; cursor: pointer; color: #FF5733;">🐶 Click here to view Google search results</a></div>'
        st.markdown(link_html, unsafe_allow_html=True)
        st.markdown('---')
        
        st.markdown("<h3 style='text-align: left; color: #4d8df2; font-size: 24px;'>Creator Details</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: left; font-size: 16px;'>Rushikesh Kothawade😄</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: left; font-size: 16px;'>Vishwakarma Institute of Information Technology, Pune 🎓</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: left; font-size: 16px;'>👨‍💻Github Link: <a href='https://github.com/RushikeshKothawade07/Dog_Vision' target='_blank'>https://github.com/RushikeshKothawade07</a></p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: left; font-size: 16px;'>🎯YouTube Channel: <a href='https://www.youtube.com/@MLTakes' target='_blank'>ML Takes</a>❤️🤑</p>", unsafe_allow_html=True)
        st.markdown('---')

def img_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

run()
