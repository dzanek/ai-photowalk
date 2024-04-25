import folium as fl
from streamlit_folium import st_folium
import streamlit as st

def get_pos(lat,lng):
    return lat,lng

params = {
    'lat': '32.7474',
    'lon': '-16.6918',
   'accuracy': 1,  # Accuracy level of the location
    'extras': 'url_m, views, geo',  # Fetch medium-sized image URLs
    'radius': 15,
    'sort': 'relevance',#'interestingness-desc',
    'per_page': 500,  # Number of photos to fetch
    'page': 1,  # Page number
    'text':'drone'
}
with st.form("my_form"):
    m = fl.Map()
    m.add_child(fl.LatLngPopup())
    map = st_folium(m, height=350, width=700)  
    params['radius'] = st.slider('How far to search?', 0, 25, 5)  
    params['accuracy'] = st.slider('How accurate location you need?', 1, 16, 4)  
    views_count = st.slider('How popular?', 0, 10, 3)  
    
    submit = st.form_submit_button('Updated the map')

if submit:
    try:
        data = get_pos(map['last_clicked']['lat'],map['last_clicked']['lng'])
    except:
        data=get_pos(params['lat'],params['lon'])
        
    if data is not None:
        print(data)
        st.write(f"Location is {data[0]}, {data[1]}")


    import flickrapi

    # Flickr API key and secret
    API_KEY = st.secrets["API_KEY"]
    API_SECRET = st.secrets["API_SECRET"]

    flickr = flickrapi.FlickrAPI(API_KEY, API_SECRET,  format='parsed-json')



    # Initialize the Flickr API
    #flickr = flickrapi.FlickrAPI(API_KEY, API_SECRET,  format='parsed-json')
    #flickr.authenticate_via_browser(perms='read')
    st.write("Search for Photos")

    # GPS coordinates
    params['lat'] = data[0]  # Latitude
    params['lon'] = data[1]  # Longitude
    lat = data[0]
    lon = data[0]
    # Parameters for the API call
    



    # Print the photo URLs
    #for photo in photos['photos']['photo']:
    #    print(photo['url_m'])
    # Make the API call
    photos = []
    for i in range(15):
        photos_json = flickr.photos.search(**params)
        st.write(f"found {len(photos_json['photos']['photo'])} photos")
        photos += ([i for i in photos_json['photos']['photo'] if int(i['views'])>10])
        if len(photos_json['photos']['photo']) == 0:
            break
        st.write(f"kept {len(photos)} photos")
        params['page'] += 1
        st.write(params['page'])
        if len(photos) > 100:
            break


    photos_urls = [photo['url_m'] for photo in photos]
   # st.write(photos_urls)
    import os
    import requests

    save_dir = f'photos_{lat}_{lon}'

    import shutil
   # st.write(os.listdir())

    shutil.rmtree(f"{save_dir}", ignore_errors=True)

    os.makedirs(save_dir, exist_ok = True)

    #st.write(os.listdir())
    st.write("Loading Photos from Flickr")
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
    from keras.applications.resnet50 import ResNet50
    from keras.preprocessing import image
    from tqdm import tqdm_notebook

    print('Load the pre-trained VGG16 model')
    #model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    print('Function to extract features from an image')
    def extract_features(img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        #img = image.load_img(img_path, target_size=(224, 224), color_mode="grayscale")
        #img_data = image.img_to_array(img)
        img_data = image.img_to_array(img)
        #st.write("data")
        #st.write(img_data[0])
        #img_data2 = np.repeat(img_data2[..., np.newaxis], 3, -1)[0]
        #img_data = np.dstack((img_data, img_data, img_data))
        #st.write(f"pseudergb")
        #st.write(img_data2[0])
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = model.predict(img_data)
        return features.flatten()
    st.write("Clustering Photos")
    # Specify the directory containing the images
    image_dir = save_dir

    # Get a list of image file paths
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    print('Extract features from all images')
    features = [extract_features(img_path) for img_path in image_paths]

    print('Cluster the images using K-Means')
    num_clusters = 6  # Specify the desired number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)

    #from sklearn.cluster import KMeans, AgglomerativeClustering
    #kmeans = AgglomerativeClustering(n_clusters=num_clusters)
    #clusters = kmeans.fit_predict(features)

    # Print the cluster assignments
   # for i, cluster in enumerate(clusters):
    #    st.write(f"Image {image_paths[i]} belongs to cluster {cluster}")


    import shutil

    for i, cluster in enumerate(clusters):
        try:
            os.mkdir(f'{save_dir}/cluster_{cluster}')
        except:
            pass
        #print(f"Image {image_paths[i]} belongs to cluster {cluster}")
        #os.popen(f'rm -r {save_dir}/cluster_{cluster}')
        shutil.copyfile(image_paths[i], f'{save_dir}/cluster_{cluster}/{image_paths[i].split("/")[-1]}')

    #from IPython.display import Image, display

 #   st.write(os.listdir('.'))
    for i in range(8):
        st.write(f"Cluster no. {i}")
        images = os.listdir(f'{save_dir}/cluster_{i}')
        #st.write(f'{save_dir}/cluster_{i}')
        st.write(f"CLuster size: {len(images)}")
        #st.write(images[:10])
        for img in images[:3]:
            #st.write(img)
            st.image(f'{save_dir}/cluster_{i}/{img}',width=300)
