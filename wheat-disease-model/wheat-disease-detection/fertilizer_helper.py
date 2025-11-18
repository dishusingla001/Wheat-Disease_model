# fertilizer_helper.py

def load_fertilizer_data():
    return {
        'Wheat_Leaf_Rust': {
            'fertilizer': 'Mancozeb 75% WP',
            'dosage': '2 kg per acre'
        },
        'Wheat_Loose_Smut': {
            'fertilizer': 'Carbendazim 50% WP',
            'dosage': '1.5 kg per acre'
        },
        'Wheat_crown_rot': {
            'fertilizer': 'Propiconazole',
            'dosage': '1 liter per acre'
        },
        'Wheat_healthy': {
            'fertilizer': 'No fertilizer needed',
            'dosage': 'N/A'
        }
    }

def get_fertilizer_info(disease_label, fertilizer_data):
    return fertilizer_data.get(disease_label, None)
