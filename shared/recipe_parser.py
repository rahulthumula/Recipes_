import os
import json
import logging
import pandas as pd
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
import asyncio
import time
import mimetypes
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.cosmos import CosmosClient
from fuzzywuzzy import fuzz
from dotenv import load_dotenv

# Local imports
from .Perplexity import PerplexityPriceEstimator
from .test import NutritionixConverter

# Load environment variables
load_dotenv()
key = os.getenv("OPENAI_API_KEY")


@dataclass
class ServingInfo:
    """Store serving-related information for a recipe."""
    servings: int
    items_per_serving: Optional[int] = None
    serving_size: Optional[str] = None
    total_yield: Optional[str] = None


class IngredientMatcher:
    """Smart ingredient matching with vector search and GPT selection."""

    def __init__(self, search_client, openai_client,user_id:str):
        self.search_client = search_client
        self.openai_client = openai_client
        self.price_scraper = PerplexityPriceEstimator(num_validations=3)
        self.user_id = user_id

        self.cosmos_container = CosmosClient(
            os.getenv("COSMOS_ENDPOINT"),
            credential=os.getenv("COSMOS_KEY")
        ).get_database_client(
            os.getenv("COSMOS_DATABASE_ID")
        ).get_container_client(
            os.getenv("COSMOS_CONTAINER_ID")
        )

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.openai_client.embeddings.create(
                    input=text.strip(),
                    model="text-embedding-ada-002"
                )
                # Return the actual embedding vector
                return response["data"][0]["embedding"]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)

        # Fallback if something goes wrong
        return []

    def check_name_similarity(self, ingredient_name: str, match_name: str) -> float:
        """Check name similarity using GPT."""
        try:
            prompt = f"""Rate the similarity between these two ingredient names based ONLY on their meaning and culinary usage.
Rules:
- Exact matches or plural forms (e.g. 'tomato' vs 'tomatoes') = 100
- Same ingredient different form (e.g. 'garlic' vs 'garlic powder') = 80
- Common substitutes (e.g. 'vegetable oil' vs 'canola oil') = 70
- Different but related items (e.g. 'cream' vs 'milk') = 50
- Different items 

Ingredient 1: {ingredient_name}
Ingredient 2: {match_name}

Return ONLY a number between 0-100. No other text."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at comparing ingredient names."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            similarity = float(response.choices[0].message.content.strip())
            return max(0, min(100, similarity)) / 100.0  # Normalize to 0-1

        except Exception:
            return 0

    def select_best_match_with_gpt(self, ingredient: Dict, matches: List[Dict]) -> Optional[Dict]:
        """
        Among a list of potential matches (filtered from vector or fuzzy search),
        use GPT to pick the single best match.
        """
        try:
            # First check name similarity for all matches
            scored_matches = []
            for match in matches:
                similarity = self.check_name_similarity(ingredient['item'], match['inventory_item'])
                # Only consider matches with >= 80% name similarity
                if similarity >= 0.8:
                    scored_matches.append({**match, 'similarity': similarity})

            if not scored_matches:
                return None

            # Format remaining matches for final selection
            matches_text = "\n".join(
                f"{idx+1}. Item: {match['inventory_item']} (Similarity: {match['similarity']*100:.1f}%)\n"
                f"   Unit: {match['measured_in']}\n"
                f"   Supplier: {match.get('supplier_name', 'Unknown')}\n"
                f"   Cost per unit: ${float(match['cost_per_unit']):.2f}"
                for idx, match in enumerate(scored_matches)
            )

            prompt = f"""Select the best matching inventory item for a recipe ingredient.
Consider ONLY:
1. Unit compatibility
2. Cost reasonableness

All these items have passed name similarity check.

Recipe Ingredient: {ingredient['item']}
Amount Needed: {ingredient['amount']} {ingredient['unit']}

Available matches:
{matches_text}

Return ONLY the number (1-{len(scored_matches)}) of the best match.
If none are suitable due to unit or cost, return 0."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at matching recipe ingredients to inventory items."},
                    {"role": "user", "content": prompt}
                ],
            )

            selected_idx_str = response.choices[0].message.content.strip()
            try:
                selected_idx = int(selected_idx_str) - 1
                if 0 <= selected_idx < len(scored_matches):
                    match = scored_matches[selected_idx]
                    return {
                        'inventory_item': match['inventory_item'],
                        'cost_per_unit': Decimal(str(match['cost_per_unit'])),
                        'unit': match['measured_in'],
                        'supplier': match.get('supplier_name', 'Unknown'),
                        'similarity': match['similarity']
                    }
                else:
                    return None
            except (ValueError, IndexError):
                return None

        except Exception as e:
            print(f"Error in select_best_match_with_gpt: {str(e)}")
            return None

    def vector_search(self, ingredient: Dict) -> Optional[Dict]:
        """
        Perform a vector similarity search using the inventory_item_vector field,
        then refine with GPT to pick the best match.
        """
        try:
            embedding = self.get_embedding(ingredient['item'])
            if not embedding:
                return None

            vector_query = VectorizedQuery(
                vector=embedding,
                k_nearest_neighbors=5,
                fields="inventory_item_vector"
            )
            results = list(self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=["inventory_item", "cost_per_unit", "measured_in", "supplier_name"],
                top=5
            ))
            if not results:
                return None

            # Convert search results to a simpler list of dicts
            matches = []
            for r in results:
                matches.append({
                    'inventory_item': r['inventory_item'],
                    'cost_per_unit': r['cost_per_unit'],
                    'measured_in': r['measured_in'],
                    'supplier_name': r.get('supplier_name', 'Unknown')
                })

            return self.select_best_match_with_gpt(ingredient, matches)
        except Exception as e:
            print(f"Vector search error for {ingredient['item']}: {str(e)}")
            return None
        
    async def add_to_cosmos(self, price_data: Dict, ingredient_name: str) -> None:
        """Add new item to user's inventory in CosmosDB."""
        try:
            # First get the user's document
            query = "SELECT * FROM c WHERE c.id = @user_id"
            params = [{"name": "@user_id", "value": self.user_id}]
            
            user_docs = list(self.cosmos_container.query_items(
                query=query,
                parameters=params,
                enable_cross_partition_query=True
            ))
            
            if not user_docs:
                print("User document not found")
                return
                
            user_doc = user_docs[0]
            
            # Create new inventory item
            new_item = {
                "Supplier Name": "Retail",
                "Inventory Item Name": ingredient_name,
                "Inventory Unit of Measure": price_data['unit'],
                "Item Name": ingredient_name,
                "Item Number": f"RTL_{ingredient_name.lower().replace(' ', '_')}",
                "Quantity In a Case": 1,
                "Measurement Of Each Item": 1,
                "Measured In": price_data['unit'],
                "Total Units": 1,
                "Case Price": float(price_data['price']),
                "Cost of a Unit": float(price_data['price']),
                "Category": "RETAIL",
                "Active": "Yes",
                "timestamp": datetime.utcnow().isoformat(),
                "Catch Weight": "N/A",
                "Priced By": "per each",
                "Splitable": "NO",
                "Split Price": "N/A"
            }
            
            # Add to user's items array
            user_doc['items'].append(new_item)
            user_doc['itemCount'] = len(user_doc['items'])
            
            # Update the document
            await self.cosmos_container.replace_item(
                item=user_doc['id'],
                body=user_doc
            )
            
            print(f"Added new inventory item: {ingredient_name}")
            
        except Exception as e:
            print(f"Error adding to CosmosDB: {str(e)}")

    def fuzzy_search(self, ingredient_name: str, threshold: int = 75) -> Optional[Dict]:
        """
        Search for an ingredient match using fuzzy string matching in user's inventory.
        """
        try:
            logging.info(f"Searching for ingredient: {ingredient_name} for user: {self.user_id}")
            
            # Query the entire user document
            query = "SELECT * FROM c WHERE c.id = @user_id"
            params = [{"name": "@user_id", "value": self.user_id}]
            
            try:
                results = list(self.cosmos_container.query_items(
                    query=query,
                    parameters=params,
                    enable_cross_partition_query=True
                ))
                
                if not results:
                    logging.warning(f"No document found for user: {self.user_id}")
                    return None
                    
                user_doc = results[0]
                if 'items' not in user_doc:
                    logging.warning(f"No items array in user document: {self.user_id}")
                    return None
                    
                # Calculate fuzzy match scores
                matches = []
                for item in user_doc['items']:
                    try:
                        if item.get('Active', 'No').lower() != 'yes':
                            continue
                            
                        inventory_name = item.get('Inventory Item Name')
                        if not inventory_name:
                            continue
                        
                        score = fuzz.ratio(ingredient_name.lower(), inventory_name.lower())
                        if score >= threshold:
                            matches.append({
                                **item,
                                'match_score': score
                            })
                    except Exception as item_error:
                        logging.error(f"Error processing item: {str(item_error)}")
                        continue
                
                # Find best match
                if matches:
                    best_match = max(matches, key=lambda x: x['match_score'])
                    return {
                        'inventory_item': best_match['Inventory Item Name'],
                        'cost_per_unit': Decimal(str(best_match['Cost of a Unit'])),
                        'unit': best_match['Measured In'],
                        'supplier': best_match['Supplier Name'],
                        'similarity': best_match['match_score'] / 100.0
                    }
                    
                return None
                
            except Exception as query_error:
                logging.error(f"Database query error: {str(query_error)}")
                return None
                
        except Exception as e:
            logging.error(f"Fuzzy search error: {str(e)}")
            return None

    async def get_best_match(self, ingredient: Dict) -> Optional[Dict]:
        """
        Get the best match from either existing inventory (via fuzzy search) or,
        failing that, estimate a retail price (via Perplexity) or do a vector search.
        """
        try:
            # 1. Try fuzzy search
            match = self.fuzzy_search(ingredient['item'])
            if match:
                return match

            # 2. If no fuzzy match found, try to get an online price
            price_data = await self.price_scraper.get_price(
                item=ingredient['item'],
                unit=ingredient['unit']
            )
            if price_data:
                total_cost = Decimal(str(price_data['price'])) * Decimal(str(ingredient['amount']))
                await self.add_to_cosmos(price_data, ingredient['item'])
                return {
                    'inventory_item': ingredient['item'],
                    'cost_per_unit': Decimal(str(price_data['price'])),
                    'unit': price_data['unit'],
                    'supplier': f"Retail ({price_data['source'].capitalize()})",
                    'is_retail_estimate': True,
                    'total_cost': total_cost
                }

            return None

        except Exception as e:
            print(f"Price estimation or best match error for {ingredient['item']}: {str(e)}")
            return None


class UnitConverter:
    """Convert between recipe and inventory units using Nutritionix API."""

    def __init__(self):
        self.nutritionix = NutritionixConverter()  # from your local .test import
        # Basic unit aliases for standardization
        self.unit_aliases = {
            'milliliter': 'ml',
            'milliliters': 'ml',
            'liter': 'l',
            'liters': 'l',
            'tablespoon': 'tbsp',
            'tablespoons': 'tbsp',
            'teaspoon': 'tsp',
            'teaspoons': 'tsp',
            'cups': 'cup',
            'gallons': 'gallon',
            'gram': 'g',
            'grams': 'g',
            'kilogram': 'kg',
            'kilograms': 'kg',
            'pound': 'lb',
            'pounds': 'lb',
            'ounce': 'oz',
            'ounces': 'oz',
            'fluid ounce': 'floz',
            'fluid ounces': 'floz',
            'fl oz': 'floz',
            'piece': 'unit',
            'pieces': 'unit',
            'whole': 'unit',
            'each': 'unit'
        }

    def _standardize_unit(self, unit: str) -> str:
        """Convert unit aliases to standard form."""
        unit = unit.lower().strip()
        return self.unit_aliases.get(unit, unit)

    def convert(self, amount: Decimal, from_unit: str, to_unit: str, ingredient: str = '') -> Decimal:
        """Convert between units using Nutritionix API if needed."""
        from_unit_std = self._standardize_unit(from_unit)
        to_unit_std = self._standardize_unit(to_unit)

        # If identical units or both are 'unit', no conversion needed
        if from_unit_std == to_unit_std or (from_unit_std == 'unit' and to_unit_std == 'unit'):
            return amount

        try:
            nutritionix_result = self.nutritionix.convert_units(
                float(amount),
                from_unit_std,
                to_unit_std,
                ingredient
            )
            if nutritionix_result and nutritionix_result.get('alt_measures'):
                # Attempt to find an exact measure match
                for measure in nutritionix_result['alt_measures']:
                    if measure['measure'].lower() == to_unit_std.lower():
                        # Example logic: (original qty * new measure weight) / measure qty
                        converted = (float(amount) * measure['serving_weight']) / measure['qty']
                        return Decimal(str(converted))

            raise ValueError(f"Could not convert {amount} {from_unit} to {to_unit} for '{ingredient}'.")
        except Exception as e:
            raise ValueError(f"Conversion error: {str(e)}")


class RecipeCostCalculator:
    """Calculate recipe costs with smart matching and conversion."""

    def __init__(self, search_client, openai_client,user_id:str):
        self.matcher = IngredientMatcher(search_client, openai_client,user_id)
        self.converter = UnitConverter()

    async def calculate_ingredient_cost(self, ingredient: Dict, match: Dict, serving_info: ServingInfo) -> Optional[Dict]:
        """
        Calculate cost info for a single ingredient. This function:
         1. Converts the recipe amount to the matched inventory unit.
         2. Multiplies converted amount by the inventory unit cost.
         3. Computes cost per serving as well.
        """
        try:
            # Add a small delay to avoid rate-limits or hammering services too quickly
            time.sleep(2)
            print(f"Calculating cost for ingredient: {ingredient}")

            recipe_amount = Decimal(str(ingredient['amount']))
            recipe_unit = ingredient['unit']
            inventory_unit = match['unit']

            # Convert the recipe amount to the inventory unit
            converted_amount = self.converter.convert(
                recipe_amount,
                recipe_unit,
                inventory_unit,
                ingredient['item']
            )

            unit_cost = Decimal(str(match['cost_per_unit']))
            total_cost = converted_amount * unit_cost

            servings_decimal = Decimal(str(serving_info.servings))
            per_serving_amount = recipe_amount / servings_decimal
            per_serving_converted = converted_amount / servings_decimal
            per_serving_cost = total_cost / servings_decimal

            return {
                'ingredient': ingredient['item'],
                'recipe_amount': f"{recipe_amount} {recipe_unit}",
                'inventory_item': match['inventory_item'],
                'inventory_unit': inventory_unit,
                'converted_amount': float(converted_amount),
                'unit_cost': float(unit_cost),
                'total_cost': float(total_cost),
                'per_serving': {
                    'amount': float(per_serving_amount),
                    'unit': recipe_unit,
                    'converted_amount': float(per_serving_converted),
                    'cost': float(per_serving_cost)
                },
                'is_retail_estimate': match.get('is_retail_estimate', False)
            }
        except Exception as e:
            print(f"Error calculating ingredient cost for {ingredient['item']}: {str(e)}")
            return None

    async def calculate_recipe_cost(self, recipe: Dict) -> Dict:
        """
        Calculate the total cost of a recipe based on its ingredients. Each ingredient
        is matched to inventory or estimated, then unit conversion is applied.
        """
        if not isinstance(recipe, dict) or 'name' not in recipe or 'ingredients' not in recipe:
            raise ValueError("Invalid recipe format")

        serving_info = ServingInfo(
            servings=recipe.get('servings', 1),
            items_per_serving=recipe.get('items_per_serving'),
            serving_size=recipe.get('serving_size'),
            total_yield=recipe.get('yield')
        )

        ingredient_costs = []
        total_cost = Decimal('0')
        print(f"Processing recipe: {recipe['name']}")
        print(f"Raw ingredients: {recipe['ingredients']}")

        for ingredient in recipe['ingredients']:
            print(f"Matching ingredient: {ingredient['item']}")
            match = await self.matcher.get_best_match(ingredient)
            if not match:
                print(f"No match found for ingredient: {ingredient['item']}")
                continue

            cost_info = await self.calculate_ingredient_cost(ingredient, match, serving_info)
            if cost_info:
                ingredient_costs.append(cost_info)
                total_cost += Decimal(str(cost_info['total_cost']))

        cost_per_serving = 0.0
        if serving_info.servings > 0:
            cost_per_serving = float(total_cost / Decimal(str(serving_info.servings)))

        return {
            'recipe_name': recipe['name'],
            'servings': serving_info.servings,
            'items_per_serving': serving_info.items_per_serving,
            'serving_size': serving_info.serving_size,
            'total_yield': serving_info.total_yield,
            'ingredients': ingredient_costs,
            'total_cost': float(total_cost),
            'cost_per_serving': cost_per_serving,
            'topping': recipe.get('topping', ''),
            'Type': recipe.get('Type', '')
        }


def extract_recipes_from_pdf(pdf_path, form_client, openai_client)-> dict:
    """Extract recipes from PDF."""
    with open(pdf_path, "rb") as doc:
        result = form_client.begin_analyze_document("prebuilt-layout", doc).result()
        content = {
            "tables": [[cell.content.strip() for cell in table.cells] for table in result.tables],
            "text": [p.content.strip() for p in result.paragraphs]
        }
    Systemprompt="""You are a precise recipe extraction specialist. Your task is to extract and standardize recipe information from any source while maintaining a consistent structure.

PRIMARY TYPE IDENTIFICATION:

MENU ITEM IDENTIFICATION:
A Menu Item MUST meet ANY of these criteria:

1. Service Specifications:
   - "served with [side/accompaniment]"
   - "garnished with [garnish]"
   - "topped with [topping]"
   - "comes with [accompaniment]"
   - "choice of [options]"

2. Portion Details:
   - Specific measurements in service units
   - Individual serving sizes
   - Exact accompaniment portions
   Examples:
   - "2 oz sauce on side"
   - "6 oz home fries"
   - "4 oz protein portion"

3. Plating Instructions:
   - Service temperature notes
   - Presentation details
   - Garnish specifications
   Examples:
   - "dust with powdered sugar"
   - "serve hot with maple syrup"
   - "garnish with fresh herbs"

4. Name Format:
   - Customer-facing descriptions
   - Size specifications
   - Service style notes
   Examples:
   - "Classic French Toast (2 slices)"
   - "Breakfast Burrito with Home Fries"
   - "16 oz Iced Caramel Latte"

RECIPE IDENTIFICATION:
A Recipe MUST meet ANY of these criteria:

1. Production Specifications:
   - Multiple serving yields
   - Bulk measurements
   - Storage instructions
   - Equipment requirements
   Examples:
   - "Yields 5 gallons"
   - "Makes 50 portions"
   - "Store in airtight container"

2. Component/Base Indicators:
   - Terms like "base," "mix," "batch"
   - Preparation instructions
   - Scaling notes
   Examples:
   - "Marinara Sauce Base"
   - "Cookie Dough Mix"
   - "Spice Blend Batch"

3. Kitchen Production Format:
   - Multiple serving yields
   - Bulk measurements
   - Component instructions
   Examples:
   - "8 servings, 2 oz each"
   - "50 portion yield"
   - "2 gallon batch"

4. Storage/Handling:
   - Shelf life information
   - Storage requirements
   - Handling instructions
   Examples:
   - "Store refrigerated"
   - "Use within 5 days"
   - "Keep below 40°F"

5.Ingredients Rules:
 -Identify whether Ingredient is a Food item or Material  

STANDARDIZATION RULES:

1. UNITS:
   Volume Measurements:
   - "ml" (milliliters) - small liquid portions
   - "l" (liters) - large liquid volumes
   - "oz" (fluid ounces) - service portions
   - "cup" (cups) - recipe measurements
   - "tbsp" (tablespoons) - recipe measurements
   - "tsp" (teaspoons) - recipe measurements
   - "gallon" (gallons) - bulk liquids
   - "cc" (cubic centimeters) - precise measurements

   Weight Measurements:
   - "g" (grams) - precise small weights
   - "kg" (kilograms) - bulk ingredients
   - "lb" (pounds) - bulk ingredients
   - "oz" (ounces) - portion weights

   Count Units:
   - "unit" - whole items
   - "slice" - portioned items
   - "piece" - component parts
   - "portion" - serving units

2. MEASUREMENT CONVERSIONS:
   Common Equivalents:
   - 1 cup = 8 oz
   - 1 tbsp = 3 tsp
   - 1 oz = 28.35 g
   - 1 lb = 16 oz
   - 1 gallon = 128 oz

   Special Measurements:
   - "pinch" = 0.125 tsp
   - "dash" = 0.0625 tsp
   - "stick of butter" = 0.5 cup
   - "small egg" = 1 unit
   - "large egg" = 1 unit
   - "clove of garlic" = 1 unit

Recipe Format:
1. Name Format:
   - Production name
   - Yield information
   - Type indicator
   Example: "Marinara Sauce Base (5 gallon yield)"

2. Batch Specifications:
   - Total yield
   - Portion size
   - Storage notes
   Example: "yields 50 portions, 4 oz each"

3. Production Instructions:
   - Equipment needs
   - Method details
   - Storage requirements
   Example: "mix in 20qt mixer, store below 40°F"

VALIDATION REQUIREMENTS:

Recipe Validation:
1. Must Have:
   - Total yield
   - Portion size
   - Storage instructions
   - Production notes
   - Scaling information
   - Equipment specifications

2. Cannot Have:
   - Single serving focus
   - Service temperatures
   - Plating instructions
   - Customer choices

OUTPUT STRUCTURE:
{
    "recipes": [
        {
            "name": "Complete Name with Size/Yield",
            "servings": number,
            "items_per_serving": number,
            "ingredients": [
                {
                    "item": "ingredient base name",
                    "amount": number,
                    "unit": "standardized unit",
                    "ItemType":"Food item or Material"
                }
            ],
            "topping": "complete topping instructions",
            "size": "recipe size or yield",
            "preparation": "recipe preparation notes",
            "Type": "ENUM['Recipe']"
        }
    ]
}

Return ONLY the JSON with no additional text or explanations."""


    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": Systemprompt},
            {"role": "user", "content": str(content)}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    logging.info(response.choices[0].message.content)
    
    try:
        recipes_data = json.loads(response.choices[0].message.content)
        return recipes_data
    except json.JSONDecodeError as e:
        raise
    except ValueError as e:
        raise

def export_to_excel(recipes_with_costs: List[Dict], output_path: Optional[str] = None) -> str:
    """Export recipe costs to Excel with detailed sheets."""
    try:
        if output_path is None:
            # Generate default output path in current directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"recipe_costs_{timestamp}.xlsx"
            
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
        os.makedirs(output_dir, exist_ok=True)
        
        with pd.ExcelWriter(output_path) as writer:
            # Recipe Summary Sheet
            summary_data = []
            for recipe in recipes_with_costs:
                summary_data.append({
                    'Recipe Name': recipe['recipe_name'],
                    'Total Cost': f"${recipe['total_cost']:.2f}",
                    'Servings': recipe['servings'],
                    'Cost per Serving': f"${recipe['cost_per_serving']:.2f}",
                    'Items per Serving': recipe.get('items_per_serving', ''),
                    'Serving Size': recipe.get('serving_size', ''),
                    'Total Yield': recipe.get('total_yield', ''),
                    'Ingredient Count': len(recipe['ingredients']),
                    'Topping': recipe['topping'],
                    'Type': recipe['Type']
                })
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Recipe Summary', index=False)
            
            # Detailed Ingredients Sheet
            ingredient_data = []
            for recipe in recipes_with_costs:
                for ing in recipe['ingredients']:
                    ingredient_data.append({
                        'Recipe Name': recipe['recipe_name'],
                        'Ingredient': ing['ingredient'],
                        'Recipe Amount': ing['recipe_amount'],
                        'Per Serving Amount': f"{ing['per_serving']['amount']:.3f} {ing['per_serving']['unit']}",
                        'Inventory Item': ing['inventory_item'],
                        'Converted Amount': f"{ing['converted_amount']:.3f} {ing['inventory_unit']}",
                        'Per Serving Converted': f"{ing['per_serving']['converted_amount']:.3f} {ing['inventory_unit']}",
                        'Unit Cost': f"${ing['unit_cost']:.3f}",
                        'Total Cost': f"${ing['total_cost']:.2f}",
                        'Cost per Serving': f"${ing['per_serving']['cost']:.2f}"
                    })
            pd.DataFrame(ingredient_data).to_excel(writer, sheet_name='Ingredient Details', index=False)
        
        return output_path
        
    except Exception as e:
        raise

def extract_recipes_from_excel(excel_path: str, openai_client) -> dict:
    """Extract recipes from Excel file."""
    try:
        # Read Excel file
        df = pd.read_excel(excel_path)
        
        # Convert DataFrame to text format similar to Form Recognizer output
        content = {
            "tables": [df.values.tolist()],  # Convert DataFrame to list of lists
            "text": df.to_string().split('\n')  # Convert to text format
        }
        
        # Use the same GPT prompt as before to maintain consistency
        system_prompt = """You are a precise recipe extraction specialist. Your task is to extract and standardize recipe information from any source while maintaining a consistent structure.

EXTRACTION RULES:
Extract all recipes from the content provided.
1. Extract ALL recipes from the provided content
2. Maintain exact measurements and units
3. Convert all text numbers to numeric values (e.g., "one" → 1)
4. Standardize ingredients to their base names
5. Capture complete procedures 

OUTPUT STRUCTURE:
Return data in this EXACT JSON format:
{
    "recipes": [
        {
            "name": "Complete Recipe Name with Size/Yield",
            "ingredients": [
                {
                    "item": "ingredient base name",
                    "amount": number,
                    "unit": "standardized unit"
                    "ItemType":"Food item or Material"
                }
            ],
            "topping": "complete topping instructions"
            "size": "recipe size or yield"
            "preparation": "recipe preparation notes"
        }
    ]
}

STANDARDIZATION RULES:
1. Units: Use these standard units ONLY:
   Volume:
   - "ml" (milliliters)
   - "l" (liters)
   - "oz" (fluid ounces)
   - "cup" (cups)
   - "tbsp" (tablespoons)
   - "tsp" (teaspoons)
   - "gallon" (gallons)
   - "cc" (cubic centimeters)
   
   Weight:
   - "g" (grams)
   - "kg" (kilograms)
   - "lb" (pounds)
   - "oz" (ounces for weight)

2. Numbers:
   - Convert all written numbers to numerals
   - Convert fractions to decimals
   - Round to 2 decimal places

3. Ingredients:
   - Use base ingredient names
   - Include preparation state in name if critical

4. Measurements:
   - Convert all measurements to standard units
   - Handle common conversions

5. Topping Instructions:
   - Include complete application method
   - Maintain sequence of steps"""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(content)}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        try:
            recipes_data = json.loads(response.choices[0].message.content)
            return recipes_data
        except json.JSONDecodeError as e:
            raise
        except ValueError as e:
            raise
            
    except Exception as e:
        raise
async def process_recipe_folder(folder_path: str, calculator: RecipeCostCalculator, form_client: DocumentAnalysisClient, openai_client: OpenAI) -> List[Dict]:
    all_recipes = []
    
    try:
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
            
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            try:
                recipes_data = None
                
                if is_excel_file(file_path):
                    recipes_data = extract_recipes_from_excel(file_path, openai_client)
                    print(f"Extracted recipes from Excel: {json.dumps(recipes_data, indent=2)}")
                    
                elif is_pdf_file(file_path) or is_image_file(file_path):
                    recipes_data = extract_recipes_from_pdf(file_path, form_client, openai_client)
                    print(f"Extracted recipes from PDF/Image: {json.dumps(recipes_data, indent=2)}")
                    
                else:
                    continue
                
                if recipes_data and recipes_data.get('recipes'):
                    for recipe in recipes_data['recipes']:
                        try:
                            print(f"Processing recipe: {recipe['name']}")
                            cost_info = await calculator.calculate_recipe_cost(recipe)
                            if cost_info:
                                print(f"Got cost info: {json.dumps(cost_info, indent=2)}")
                                all_recipes.append(cost_info)
                            else:
                                print(f"No cost info generated for recipe: {recipe['name']}")
                        except Exception as e:
                            print(f"Error calculating recipe cost: {str(e)}")
                            continue
            
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue
                
        return all_recipes
        
    except Exception as e:
        print(f"Error in process_recipe_folder: {str(e)}")
        return []
def is_image_file(file_path: str) -> bool:
    """Check if file is an image based on its mimetype."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type is not None and mime_type.startswith('image/')

def is_pdf_file(file_path: str) -> bool:
    """Check if file is a PDF."""
    return file_path.lower().endswith('.pdf')

def is_excel_file(file_path: str) -> bool:
    """Check if file is an Excel file."""
    return file_path.lower().endswith(('.xlsx', '.xls'))    


async def main():
    """Run the recipe cost calculator on a folder of files."""
    try:
        # Setup clients
        search_client = SearchClient(
            endpoint=os.getenv("AZURE_AISEARCH_ENDPOINT"),
            index_name="drift-customer",
            credential=AzureKeyCredential(os.getenv("AZURE_AISEARCH_APIKEY"))
        )
        
        form_client = DocumentAnalysisClient(
            endpoint=os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_FORM_RECOGNIZER_KEY"))
        )
        
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        user_id = "user123"  # Update
        # Initialize calculator
        calculator = RecipeCostCalculator(search_client, openai_client,user_id)
        
        # Process recipe folder
        #"C:\Users\rahul\Downloads\Bulk Recipes and Sub-recipes-20241205T074727Z-001"
        folder_path = "C:/Users/rahul/Downloads/New added menu items 11-26-2024-20241205T074732Z-001/New added menu items 11-26-2024"  # Update this path
        recipe_costs = await process_recipe_folder(folder_path, calculator, form_client, openai_client)
        
        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(folder_path, f"recipe_costs_{timestamp}.xlsx")
        
        # Export results
        final_path = export_to_excel(recipe_costs, output_path)
        print(f"Results saved to: {final_path}")
        
    except Exception as e:
        raise

if __name__ == "__main__":
    asyncio.run(main())