import torch
import numpy as np

from tqdm import tqdm

from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
from global_land_mask import globe

from geopy.geocoders import Nominatim
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
    for (coordinates, distance) in best_coordinates:
        result.append(coordinates)

    return result

print(find_best(5))

# def coordinates_to_continent(lat, long):
#     try: 
#         country = geolocator.reverse([lat, long]).raw['address']['country']
#     except:
#         country = 'Unknown'
#     try:
#         cn_a2_code =  country_name_to_country_alpha2(country)
#     except:
#         cn_a2_code = 'Unknown' 
#     try:
#         cn_continent = country_alpha2_to_continent_code(cn_a2_code)
#     except:
#         cn_continent = 'Unknown' 
#     return cn_continent

# def find_best_per_continent(continent):
#     best = 500
#     for lat in tqdm(range(-90, 91)):
#         for long in range(-180, 181):
#             if globe.is_land(lat, long):
#                 if coordinates_to_continent(lat, long) == continent:
#                     disasters = predict(lat, long)
#                     if disasters < best:
#                         best = disasters

#     ret = []
#     for lat in tqdm(range(-90, 91)):
#         for long in range(-180, 181):
#             if globe.is_land(lat, long):
#                 if coordinates_to_continent(lat, long) == continent:
#                     disasters = predict(lat, long)
#                     if disasters == best:
#                         ret = [lat, long]
#     return ret