from flask import Flask, request
from flask.templating import render_template
import pickle, logging, sys
import pandas as pd
import numpy as np

app = Flask(__name__)

# setup basic logging with time, log level & log message
logger = logging.getLogger()
logging.basicConfig(
    stream=sys.stdout,  # uncomment to redirect output to console
    # filename='scraping.log',
    format="%(asctime)s %(message)s",
    # filemode='w',
    level=logging.DEBUG
)

logging.info('Loading model...')
model = pickle.load(open('./model_final.sav', 'rb'))
logging.info('Loading dataset web...')
df_web = pd.read_excel('./dataset_web.xlsx')
logging.info('Loading dataset matching...')
df_match = pd.read_excel('./dataset_matching.xlsx')

logging.info('Ready!')

furniture_cat = model.named_steps['preprocessor'].transformers_[0][1].categories_[0]
location_cat = model.named_steps['preprocessor'].transformers_[0][1].categories_[1]

@app.route('/', methods=['GET', 'POST'])
def home():
    while True:
        carousel_df = df_web[['link','img_link']].sample(n=3)

        for i in carousel_df['img_link']:
            if type(i) == float:
                continue
        break

    carousel_link = carousel_df['link'].tolist()
    carousel_image = carousel_df['img_link'].tolist()
    # top_carousel = '' # ========== REMOVE THIS IN PRODUCTION ==========!!!!!!!!!!!!!!!!!!!!!!!!
    return render_template('index.html', furniture_dict=furniture_cat, location_dict=location_cat, carousel_link=carousel_link, carousel_image=carousel_image)

@app.route('/predict_calc')
def predict():
    kamarT = int(request.args.get('x1'))
    kamarM = int(request.args.get('x2'))
    luasA = int(request.args.get('x3'))
    luasB = int(request.args.get('x4'))
    garasi = int(request.args.get('x5'))
    listrik = int(request.args.get('x6'))
    perabotan = request.args.get('x7')
    lokasi = request.args.get('x8')
    dict = {'bed': [kamarT],
            'bath': [kamarM],
            'land_size': [luasA],
            'building_size': [luasB],
            'garage': [garasi],
            'electricity': [listrik],
            'furniture': [perabotan],
            'location': [lokasi]}
    payload = pd.DataFrame(dict)
    result = model.predict(payload)
    return([to_rupiah(result[0] * 0.9), to_rupiah(result[0] * 1.1), str(result[0])]) # return harga +- 10% error and raw pred

@app.route('/recommendation')
def recommendation():
    price = float(request.args.get('x0'))
    kamarT = int(request.args.get('x1'))
    kamarM = int(request.args.get('x2'))
    luasA = int(request.args.get('x3'))
    luasB = int(request.args.get('x4'))
    garasi = int(request.args.get('x5'))
    listrik = int(request.args.get('x6'))
    perabotan = request.args.get('x7')
    lokasi = request.args.get('x8')
    dict = {'price': [price],
            'bed': [kamarT],
            'bath': [kamarM],
            'land_size': [luasA],
            'building_size': [luasB],
            'garage': [garasi],
            'electricity': [listrik],
            'furniture': [perabotan],
            'location': [lokasi]}
    payload = pd.DataFrame(dict)
    encoded_cols = model.named_steps['preprocessor'].transformers_[0][1].transform(payload.select_dtypes('object'))[0]

    payload['furniture'] = encoded_cols[0]
    payload['location'] = encoded_cols[1]

    def custom_cdist(df1, df2):
        # Convert dataframes to numpy arrays for efficient computation
        arr1 = df1.values
        arr2 = df2.values

        # Initialize the distance matrix
        distances = np.zeros((arr1.shape[0], arr2.shape[0]))

        # Compute pairwise distances
        for i in range(arr1.shape[0]):
            for j in range(arr2.shape[0]):
                distances[i, j] = np.linalg.norm(arr1[i] - arr2[j])
        
        return distances

    dist_matrix = custom_cdist(df_match, payload)

    df_pred = df_web.iloc[pd.DataFrame(dist_matrix, columns=['distance']).sort_values(by=['distance'])[0:6].index]
    df_pred['price'] = df_pred['price'].map(lambda x: to_rupiah(x))
    df_pred['img_link'] = df_pred['img_link'].fillna("https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg")

    return df_pred.reset_index().to_dict()

def to_rupiah(value):
    str_value = str(value)
    separate_decimal = str_value.split(".")
    # after_decimal = separate_decimal[0]

    reverse = separate_decimal[0][::-1]
    temp_reverse_value = ""

    for index, val in enumerate(reverse):
        if (index + 1) % 3 == 0 and index + 1 != len(reverse):
            temp_reverse_value = temp_reverse_value + val + "."
        else:
            temp_reverse_value = temp_reverse_value + val

    temp_result = temp_reverse_value[::-1]

    return "Rp " + temp_result

if __name__ == "__main__":
    app.run()
