# Weather Description Neural Network


Neural network to create weather descriptions given an image and current weather measurements.

### Data Sets:
* Images: Kat Kam (private)
* Weather Data: GHCN YVR Airport

### Dependencies:
* Python 3
* Numpy
* Pandas
* Scikit Learn
* OpenCV

### How to run:
1. Clone the repo. 
```
git clone https://github.com/bwconrad/Weather-Description-Generator.git
```
2. In the project directory run the program with the arguments [images directoy] [weather data directory] [output CSV]. 
```
python3 weather_imgs.py Data\katkam-scaled Data\yvr-weather results.csv
```
3. The program will output the accuracy on the testing data and the predictions are saved as the inputted CSV filename.

## Results
Network achieves 70% accuracy using a 75:25 data split and training for 3000 epoches.


