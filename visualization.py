import numpy as np
import matplotlib.pyplot as plt
from extraction import average_yearly_temperatures


EARTH_RADIUS = 6371
p = 6


def predict_temperatures(temps, latitudes, longitudes):
    lat1_r = temps[:, 0] / 180 * np.pi
    lon1_r = temps[:, 1] / 180 * np.pi

    lat2_r = latitudes / 180 * np.pi
    lon2_r = longitudes / 180 * np.pi

    sin_lat1 = np.sin(lat1_r)
    cos_lat1 = np.cos(lat1_r)

    sin_lat2 = np.sin(lat2_r)
    cos_lat2 = np.cos(lat2_r)

    predicted = np.empty(latitudes.size)

    for i in range(latitudes.size):
        print(i)
        distances = np.empty(temps.shape[0])

        same = (lat1_r == lat2_r[i]) & (lon1_r == lon2_r[i])
        distances[same] = 0

        antipodes = ~same & (lat1_r == -lat2_r[i]) & ((lon1_r == lon2_r[i] + np.pi) | (lon1_r == lon2_r[i] - np.pi))
        distances[antipodes] = EARTH_RADIUS * np.pi

        normal = ~same & ~antipodes
        distances[normal] = EARTH_RADIUS * np.arccos(sin_lat1[normal] * sin_lat2[i] + cos_lat1[normal]
                                                     * cos_lat2[i] * np.cos(lon2_r[i] - lon1_r[normal]))

        if (distances < 1).any():
            predicted[i] = temps[np.argmin(distances), 2]
        else:
            weights = 1 / np.power(distances, p)
            predicted[i] = np.sum(temps[:, 2] * weights) / weights.sum()

    return predicted


def visualize(year, temps):
    width = 360
    height = 180

    latitudes = np.tile(np.arange(90, -90, -1)[np.newaxis].T, width).flatten()
    longitudes = np.tile(np.arange(-180, 180), height)

    image = predict_temperatures(temps, latitudes, longitudes).reshape((height, width))

    fig, ax = plt.subplots()
    ax.imshow(image)
    fig.savefig(f"{year}.png")


if __name__ == "__main__":
    temperatures = average_yearly_temperatures(2015)
    visualize(2015, temperatures)
    plt.show()
