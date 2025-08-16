from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

CROP_FACTS = {
    'rice': 'Rice is the staple food for over half of the world\'s population.',
    # ... (keep all the other facts here) ...
    'coffee': 'Brazil is the world\'s largest producer of coffee beans.'
}

# Load the trained model and the NEW crop ranges data
model = joblib.load('crop_model.joblib')
crop_ranges_df = pd.read_csv('crop_ranges.csv').set_index('label') # Load the new ranges file
CROP_NAMES = sorted(crop_ranges_df.index.unique())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # This part remains the same
        data = [float(request.form[key]) for key in ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']]
        probabilities = model.predict_proba([data])[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_crops = model.classes_[top_3_indices]
        top_3_probs = probabilities[top_3_indices]

        results = []
        for crop_name, prob in zip(top_3_crops, top_3_probs):
            stats = crop_ranges_df.loc[crop_name] # Use the ranges dataframe
            desc_text = (
                f"Thrives with nitrogen from {int(stats['N_min'])} to {int(stats['N_max'])}. "
                f"Prefers a temperature of {stats['temperature_min']:.1f}°C to {stats['temperature_max']:.1f}°C and rainfall from {int(stats['rainfall_min'])} to {int(stats['rainfall_max'])} mm."
            )
            results.append({
                'name': crop_name.capitalize(),
                'probability': f"{prob*100:.2f}%",
                'description': desc_text,
                'image_url': f"/static/images/{crop_name.lower()}.jpg",
                'fact': CROP_FACTS.get(crop_name.lower(), "No fun fact available.")
            })
        return render_template('result.html', predictions=results)

@app.route('/reverse_lookup', methods=['GET', 'POST'])
def reverse_lookup():
    if request.method == 'POST':
        selected_crop = request.form['crop_name']
        conditions = crop_ranges_df.loc[selected_crop].to_dict() # Get ranges
        conditions['label'] = selected_crop
        return render_template('reverse_lookup.html', crops=CROP_NAMES, conditions=conditions, crop_selected=True)
        
    return render_template('reverse_lookup.html', crops=CROP_NAMES, crop_selected=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)