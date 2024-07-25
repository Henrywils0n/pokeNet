import requests

def get_pokemon_species_names(generation_id):
    url = f"https://pokeapi.co/api/v2/generation/{generation_id}/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        species = data['pokemon_species']
        names = [species_data['name'].capitalize() for species_data in species]
        return names
    else:
        return []