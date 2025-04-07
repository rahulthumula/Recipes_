import requests
from decimal import Decimal
import json
from typing import Dict, Optional

class NutritionixConverter:
    def __init__(self):
        self.api_headers = {
            "x-app-id": "10e8bede",
            "x-app-key": "62dc80999139cdad315d46a83d365af8",
            "x-remote-user-id": "0"
        }
        self.api_url = "https://trackapi.nutritionix.com/v2"

    def get_conversion_data(self, amount: float, from_unit: str, item: str) -> Optional[Dict]:
        """Get conversion data for a specific amount and unit of an ingredient"""
        try:
            # Format query
            query = f"{amount} {from_unit} {item}"
            print(f"\nTesting conversion for: '{query}'")
            
            response = requests.post(
                f"{self.api_url}/natural/nutrients",
                headers=self.api_headers,
                json={"query": query}
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get('foods'):
                print("No food data found")
                return None
                
            food = data['foods'][0]
            
            # Extract useful conversion information
            conversion_data = {
                'query_amount': amount,
                'query_unit': from_unit,
                'item': item,
                'serving_weight_grams': food.get('serving_weight_grams'),
                'serving_unit': food.get('serving_unit'),
                'serving_qty': food.get('serving_qty'),
                'alt_measures': food.get('alt_measures', [])
            }
            
            return conversion_data
            
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return None

    def convert_units(self, amount: float, from_unit: str, to_unit: str, item: str) -> Optional[Dict]:
        """Convert between different units for a specific ingredient"""
        conversion_data = self.get_conversion_data(amount, from_unit, item)
        if not conversion_data:
            return None
            
        print("\nConversion Data:")
        print(f"Original: {amount} {from_unit} {item}")
        print(f"Weight in grams: {conversion_data['serving_weight_grams']}g")
        
        print("\nAvailable conversions:")
        for measure in conversion_data['alt_measures']:
            print(f"- {measure['qty']} {measure['measure']} = {measure['serving_weight']}g")
            
        return conversion_data

def test_recipe_conversions():
    converter = NutritionixConverter()
    
    # Test cases relevant to your recipe
    test_cases = [
        {
            'amount': 3,
            'from_unit': 'tablespoon',
            'to_unit': 'oz',
            'item': 'vanilla extract'
        },
        {
            'amount': 12,
            'from_unit': 'unit',
            'to_unit': 'count',
            'item': 'large eggs'
        },
        {
            'amount': 3,
            'from_unit': 'tablespoon',
            'to_unit': 'oz',
            'item': 'ground cinnamon'
        },
        {
            'amount': 2,
            'from_unit': 'slice',
            'to_unit': 'oz',
            'item': 'brioche bread'
        }
    ]
    
    print("Starting Recipe Conversion Tests...")
    
    for test in test_cases:
        print("\n" + "="*50)
        result = converter.convert_units(
            test['amount'],
            test['from_unit'],
            test['to_unit'],
            test['item']
        )
        
        if result:
            print("\nConversion Summary:")
            print(f"Item: {test['item']}")
            print(f"From: {test['amount']} {test['from_unit']}")
            print(f"To: {test['to_unit']}")
            if result['alt_measures']:
                relevant_measures = [m for m in result['alt_measures'] 
                                   if m['measure'].lower() == test['to_unit'].lower()]
                if relevant_measures:
                    conversion = relevant_measures[0]
                    converted_amount = (test['amount'] * conversion['serving_weight']) / conversion['qty']
                    print(f"Converted amount: {converted_amount:.2f}g")
        
        input("\nPress Enter to continue to next test...")

if __name__ == "__main__":
    test_recipe_conversions()