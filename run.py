import torch
import numpy as np

from tqdm import tqdm

from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
from global_land_mask import globe

from geopy.geocoders import Nominatim

import webbrowser

geolocator = Nominatim(user_agent="wolfhacks")

from train import Model

# load the trained model
model = Model(2, 1)
model = torch.load("model.pt")

# helper predict function that predicts how much natural disasters will happen at a certain longitude and latitude
def predict(longitude, latitude):
    with torch.no_grad():
        model.eval()
        data = torch.from_numpy(np.array([longitude, latitude], np.float32))
        pred = model(data)
        return pred.item()

# find the best place to settle
def find_best(top_n=1):
    best_coordinates = []
    for lat in tqdm(range(-90, 91)):
        for long in range(-180, 181):
            if globe.is_land(lat, long):
                disasters = predict(lat, long)
                best_coordinates.append(([lat, long], disasters))
    best_coordinates.sort(key=lambda x : x[1])
    best_coordinates = best_coordinates[:top_n]

    result = []
    for (coordinates, _) in best_coordinates:
        result.append(coordinates)

    return result

result = find_best()
lat, long = result[0]
print(lat, long)
webbrowser.open(f"https://www.google.com/maps/place/{abs(lat)}%C2%B000'00.0%22{'S' if lat <= 0 else 'N'}+{abs(long)}%C2%B000'00.0%22{'W' if long <= 0 else 'E'}/@{lat},{long},17z/")