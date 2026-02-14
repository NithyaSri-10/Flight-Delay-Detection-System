import requests
import os
import random
from datetime import datetime
from typing import Dict, Optional

class WeatherService:
    """Fetches and analyzes weather data for airports (real or simulated)."""
    
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.cache = {}

        # Predefined coordinates for common U.S. airports
        self.airport_coords = {
            'ATL': (33.6407, -84.4277),
            'LAX': (33.9425, -118.4081),
            'ORD': (41.9742, -87.9073),
            'DFW': (32.8975, -97.0382),
            'DEN': (39.8561, -104.6737),
            'JFK': (40.6413, -73.7781),
            'SFO': (37.6213, -122.3790),
            'SEA': (47.4502, -122.3088),
            'LAS': (36.0840, -115.1537),
            'MIA': (25.7959, -80.2870)
        }

    # ----------------------------------------------------------------------
    def get_weather_by_airport(self, airport_code: str) -> Optional[Dict]:
        """
        Get live or simulated weather data for a given airport.
        """
        airport_code = airport_code.upper().strip()
        if airport_code not in self.airport_coords:
            print(f"⚠️ Unknown airport code: {airport_code}")
            return None

        # Return cached data if available (avoid spamming API)
        if airport_code in self.cache and (
            datetime.now() - datetime.fromisoformat(self.cache[airport_code]['timestamp'])
        ).seconds < 300:
            return self.cache[airport_code]

        lat, lon = self.airport_coords[airport_code]

        # Try live API only if API key is available
        if self.api_key and self.api_key.lower() != "demo-key":
            try:
                params = {
                    'lat': lat,
                    'lon': lon,
                    'appid': self.api_key,
                    'units': 'metric'
                }
                response = requests.get(self.base_url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    weather_info = {
                        'airport': airport_code,
                        'temperature': data['main']['temp'],
                        'humidity': data['main']['humidity'],
                        'wind_speed': data['wind'].get('speed', 0),
                        'cloudiness': data['clouds'].get('all', 0),
                        'precipitation': data.get('rain', {}).get('1h', 0),
                        'conditions': data['weather'][0]['main'],
                        'timestamp': datetime.now().isoformat()
                    }
                    self.cache[airport_code] = weather_info
                    print(f"✅ Fetched live weather for {airport_code}")
                    return weather_info
                else:
                    print(f"⚠️ OpenWeather API error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"❌ Weather API error for {airport_code}: {e}")

        # ------------------------------------------------------------------
        # DEMO MODE: Generate realistic random weather data
        print(f"🌤 Using simulated weather data for {airport_code}")
        simulated_weather = {
            'airport': airport_code,
            'temperature': round(random.uniform(10, 35), 1),
            'humidity': random.randint(40, 90),
            'wind_speed': round(random.uniform(1, 10), 1),
            'cloudiness': random.randint(10, 90),
            'precipitation': round(random.uniform(0, 5), 1),
            'conditions': random.choice(["Clear", "Clouds", "Rain", "Fog", "Drizzle"]),
            'timestamp': datetime.now().isoformat()
        }
        self.cache[airport_code] = simulated_weather
        return simulated_weather

    # ----------------------------------------------------------------------
    def get_weather_impact_on_delay(self, weather: Dict) -> Dict:
        """
        Compute a weather risk factor for potential flight delays.
        """
        if not weather:
            return {'risk_factor': 0, 'impact': 'No weather data available'}

        risk_factor = 0
        impacts = []

        # Temperature
        temp = weather['temperature']
        if temp < -10 or temp > 35:
            risk_factor += 0.2
            impacts.append(f"Extreme temperature ({temp}°C)")

        # Wind
        wind = weather['wind_speed']
        if wind > 25:
            risk_factor += 0.3
            impacts.append(f"High wind speed ({wind} m/s)")

        # Rain / Snow
        precip = weather['precipitation']
        if precip > 5:
            risk_factor += 0.35
            impacts.append(f"Heavy precipitation ({precip} mm)")

        # Clouds
        clouds = weather['cloudiness']
        if clouds > 80:
            risk_factor += 0.1
            impacts.append(f"Heavy cloud coverage ({clouds}%)")

        # Conditions
        condition = weather['conditions'].lower()
        if 'thunderstorm' in condition:
            risk_factor += 0.4
            impacts.append("Thunderstorm conditions")
        elif 'snow' in condition:
            risk_factor += 0.35
            impacts.append("Snow conditions")
        elif 'fog' in condition:
            risk_factor += 0.25
            impacts.append("Fog conditions")

        risk_factor = min(risk_factor, 1.0)

        return {
            'risk_factor': round(risk_factor, 2),
            'impacts': impacts,
            'weather_conditions': weather['conditions'],
            'temperature': weather['temperature'],
            'wind_speed': weather['wind_speed'],
            'precipitation': weather['precipitation']
        }


# ----------------------------------------------------------------------
# Initialize a global instance
# ----------------------------------------------------------------------
weather_service = WeatherService()
