import folium as fl
from streamlit_folium import st_folium
import streamlit as st

def get_pos(lat,lng):
    return lat,lng

m = fl.Map()

m.add_child(fl.LatLngPopup())

map = st_folium(m, height=350, width=700)


data = get_pos(map['last_clicked']['lat'],map['last_clicked']['lng'])

if data is not None:
    st.write(data)


import flickrapi

# Flickr API key and secret
API_KEY = u'1a30984068369e49e61be918aa2e0e07'
API_SECRET = u'ecab870fba6384eb'

# Initialize the Flickr API
flickr = flickrapi.FlickrAPI(API_KEY, API_SECRET,  format='parsed-json')
flickr.authenticate_via_browser(perms='read')

# GPS coordinates
lat = data[0]  # Latitude
lon = data[1]  # Longitude

# Parameters for the API call
params = {
    'lat': lat,
    'lon': lon,
#    'accuracy': 16,  # Accuracy level of the location
    'extras': 'url_m, views, geo',  # Fetch medium-sized image URLs
    'radius': 2,
    'sort': 'interestingness-desc',
    'per_page': 500,  # Number of photos to fetch
    'page': 1,  # Page number
    'tags':'landscape, street, sunset, art, portrait'
}

# Make the API call
photos_json = flickr.photos.search(**params)

# Print the photo URLs
#for photo in photos['photos']['photo']:
#    print(photo['url_m'])

print(len(photos_json['photos']['photo']))
photos = [i for i in photos_json['photos']['photo'] if int(i['views'])>1000]
photos_urls = [photo['url_m'] for photo in photos]
print(len(photos))

import os
import requests

save_dir = f'photos_{lat}_{lon}'
os.makedirs(save_dir, exist_ok = True)

for p in photos_urls:
    response = requests.get(p, stream=True)
    if response.status_code == 200:
    # Open a file for writing in binary mode
        with open(f'{save_dir}/{p.split("/")[-1]}', 'wb') as file:
            # Iterate over the response data in chunks
            for chunk in response.iter_content(chunk_size=1024):
                # Write the chunk to the file
                file.write(chunk)
        #print(f'File downloaded and saved to {save_dir}')


save_dir = f'photos_{lat}_{lon}'

import os
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from tqdm import tqdm_notebook

print('Load the pre-trained VGG16 model')
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

print('Function to extract features from an image')
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# Specify the directory containing the images
image_dir = save_dir

# Get a list of image file paths
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

print('Extract features from all images')
features = [extract_features(img_path) for img_path in image_paths]

print('Cluster the images using K-Means')
num_clusters = 8  # Specify the desired number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(features)

from sklearn.cluster import KMeans, AgglomerativeClustering
kmeans = AgglomerativeClustering(n_clusters=num_clusters)
clusters = kmeans.fit_predict(features)

# Print the cluster assignments
for i, cluster in enumerate(clusters):
    print(f"Image {image_paths[i]} belongs to cluster {cluster}")


import shutil

for i, cluster in enumerate(clusters):
    try:
      os.mkdir(f'{save_dir}/cluster_{cluster}')
    except:
      pass
    #print(f"Image {image_paths[i]} belongs to cluster {cluster}")
    #os.popen(f'rm -r {save_dir}/cluster_{cluster}')
    shutil.copyfile(image_paths[i], f'{save_dir}/cluster_{cluster}/{image_paths[i].split("/")[-1]}')

from IPython.display import Image, display



for i in range(8):
  print(i)
  images = os.listdir(f'{save_dir}/cluster_{i}')
  for img in images[:3]:
    display(Image(filename=f'{save_dir}/cluster_{i}/{img}', width=200))
