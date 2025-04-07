import os
import json
import logging
import asyncio
import statistics
from typing import Dict, List, Optional
from decimal import Decimal
from dotenv import load_dotenv
from openai import OpenAI

class PerplexityPriceEstimator:
    """Wholesale price estimation system with enhanced prompting."""
    SYSTEM_PROMPT = """You are a wholesale food pricing database specializing in accurate, current US wholesale food prices for food service businesses. 
IMPORTANT: Return EXACTLY this JSON format and nothing else:
{
    "price": [plain number between 0.01 and 100.00],
    "source": "Wholesale Database"
}

Examples of CORRECT responses:
{
    "price": 5.52,
    "source": "Wholesale Database"
}
{
    "price": 0.62,
    "source": "Wholesale Database"
}

Examples of INCORRECT responses:
{"price": "$5.52", "source": "Wholesale Database"}  // No currency symbols
{"price": "5.52", "source": "Wholesale Database"}   // No string numbers
{"price": 5.52}  // Missing source field
{price: 5.52, source: "Wholesale Database"}  // Missing quotes

Rules:
1. Price must be:
   - A plain number (5.52)
   - No currency symbols ($)
   - No quotation marks
   - No commas
   - Between 0.01 and 100.00
   - Rounded to 2 decimal places

2. Format must be:
   - Valid JSON
   - Double quotes (not single)
   - Exactly as shown above
   - No additional fields
   - No missing fields
Important Guidelines- The price must be a plain number (no currency symbols, no commas)
2. Price Guidelines:
- Provide prices in USD per specified unit
- Use standard US food service wholesale prices
- Consider current market rates
- Assume bulk purchasing (food service quantities)
- Price for commercial/food service quality
- All prices must be wholesale/bulk rates, not retail
- Round to 2 decimal places
- Prices must be between $0.01 and $100.00 per unit

3. Wholesale Context:
- Prices should reflect food service supplier rates
- Account for standard bulk discounts
- Use food service distribution pricing
- Consider regional market averages
- Base on standard case/bulk quantities
- Price as if ordering recurring bulk supply

5. Response Requirements:
- Return ONLY the JSON object
- No explanations or additional text
- No markdown formatting
- No conversational elements"""

    USER_PROMPT = """Provide the current wholesale price for 1 {unit} of {item} following these rules:

1. Return only this JSON format:
{{
    "price": number,
    "source": "Wholesale Database"
}}

2. Consider:
- Standard wholesale/bulk pricing
- Current market rates
- Food-service quality
- Standard US market
- {specific_guidelines}

3. Units: Price should be per 1 {unit}
{unit_note}

Remember: Return ONLY the JSON object with no additional text."""

    def __init__(self, num_validations: int = 1):
        """Initialize with number of validation calls."""
        load_dotenv()
        self.api_key = os.environ.get("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("Perplexity API key not found")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.perplexity.ai"
        )
        
        self.num_validations = num_validations
        self.categories = {
            'proteins': {
                'base_price': Decimal('2.00'),
                'keywords': ['meat', 'beef', 'chicken', 'pork', 'fish', 'seafood', 'lamb', 'turkey', 
                           'sausage', 'tofu', 'bacon', 'protein', 'eggs']
            },
            'dairy': {
                'base_price': Decimal('0.75'),
                'keywords': ['milk', 'cheese', 'cream', 'yogurt', 'butter', 'dairy']
            },
            'produce': {
                'base_price': Decimal('0.50'),
                'keywords': ['vegetable', 'fruit', 'produce', 'fresh', 'salad', 'herb', 'lettuce', 
                           'tomato', 'onion', 'carrot', 'potato', 'berry', 'apple', 'grape']
            },
            'grains': {
                'base_price': Decimal('0.25'),
                'keywords': ['bread', 'rice', 'pasta', 'flour', 'grain', 'cereal', 'oat', 'cracker', 
                           'tortilla', 'wrap', 'brioche']
            },
            'condiments': {
                'base_price': Decimal('0.50'),
                'keywords': ['sauce', 'dressing', 'oil', 'vinegar', 'spread', 'dip', 'condiment', 
                           'seasoning', 'spice', 'marinade', 'cinnamon']
            },
            'baking': {
                'base_price': Decimal('0.30'),
                'keywords': ['sugar', 'baking', 'chocolate', 'cocoa', 'vanilla', 'yeast', 'powder']
            },
            'default': {
                'base_price': Decimal('0.50'),
                'keywords': []
            }
        }
        
        self.logger = logging.getLogger('price_estimator')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _get_category_price(self, item: str) -> Decimal:
        """Get base price from category."""
        item = item.lower()
        for category, info in self.categories.items():
            if any(keyword in item for keyword in info['keywords']):
                return info['base_price']
        return self.categories['default']['base_price']
    
    def _get_specific_guidelines(self, item: str, unit: str) -> tuple[str, str]:
        item = item.lower()
        unit = unit.lower()
        
        # Standard units mapping
        standard_units = {
            "eggs": {"default_unit": "dozen", "conversion": {"": 12, "unit": 1/12}},
            "vanilla": {"default_unit": "tbsp", "typical_price": 0.62},
            "cinnamon": {"default_unit": "tbsp", "typical_price": 0.25},
            "bread": {"default_unit": "slice", "typical_price": 0.50},
        }
        
        # Get category base price
        category_price = self._get_category_price(item)
        
        # Dictionary of item categories and their guidelines
        guidelines = {
            "bread": {
                "guidelines": f"Standard commercial bakery wholesale pricing\nBulk food service quantities\nPriced for recurring food service orders\nTypical range: ${float(category_price):.2f}-${float(category_price * 2):.2f} per slice",
                "unit_note": "Price per slice for food service"
            },
            "eggs": {
                "guidelines": f"Standard commercial grade eggs\nBulk food service pricing\nTypical range: ${float(category_price):.2f}-${float(category_price * 2):.2f} per unit",
                "unit_note": "Convert to standard unit pricing if needed"
            },
            "vanilla": {
                "guidelines": "Food service grade vanilla extract\nBulk liquid measurement\nStandard food service packaging",
                "unit_note": "Price per tablespoon of extract"
            },
            "cinnamon": {
                "guidelines": "Ground cinnamon\nBulk food service grade\nStandard packaging",
                "unit_note": "Price per tablespoon of ground spice"
            },
            "brioche": {
                "guidelines": "Premium bakery wholesale pricing\nBulk food service quantities\nPriced for recurring orders",
                "unit_note": "Price per slice for food service"
            }
        }

        # Find matching item
        matched_item = None
        for known_item in guidelines.keys():
            if known_item in item or item in known_item:
                matched_item = known_item
                break

        if matched_item:
            return guidelines[matched_item]["guidelines"], guidelines[matched_item]["unit_note"]
        
        # Default category-based guidelines
        for category, info in self.categories.items():
            if any(keyword in item for keyword in info['keywords']):
                return (
                    f"Standard wholesale food service pricing\nBulk quantities\n{category.title()} category pricing",
                    f"Price per {unit} for food service"
                )
        
        return (
            "Standard wholesale food service pricing\nCommercial grade product",
            f"Price per {unit} of product"
        )

    async def _make_api_call(self, item: str, unit: str) -> Optional[float]:
        """Make single API call with enhanced prompting."""
        try:
            specific_guidelines, unit_note = self._get_specific_guidelines(item, unit)
            
            formatted_user_prompt = self.USER_PROMPT.format(
                unit=unit.lower().strip(),
                item=item.lower().strip(),
                specific_guidelines=specific_guidelines,
                unit_note=unit_note
            )

            response = self.client.chat.completions.create(
                model="sonar",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": formatted_user_prompt}
                ],
                temperature=0.1,
                max_tokens=15000,
                top_p=0.1
            )
            
            if not response.choices:
                return None
                
            result = self._extract_json(response.choices[0].message.content)
            
            if result and "price" in result:
                try:
                    price = float(result["price"])
                    # Add reasonable bounds checking
                    if 0.01 <= price <= 100.0:
                        return price
                    else:
                        self.logger.warning(f"Price {price} outside reasonable bounds for {item}")
                        return None
                except (ValueError, TypeError):
                    return None
                    
            return None
                
        except Exception as e:
            self.logger.error(f"Error making API call: {str(e)}")
            return None

    async def _get_multiple_prices(self, item: str, unit: str) -> List[float]:
        """Get multiple price estimates."""
        prices = []
        for i in range(self.num_validations):
            try:
                price = await self._make_api_call(item, unit)
                if price is not None:
                    prices.append(price)
                await asyncio.sleep(0.5)  # Small delay between calls
            except Exception as e:
                self.logger.error(f"Error in validation call {i+1}: {str(e)}")
        return prices

    def _extract_json(self, content: str) -> Optional[Dict]:
        """Extract and validate JSON from response with enhanced cleanup."""
        try:
            self.logger.debug(f"Raw content: {content}")
            
            # Enhanced content cleanup
            content = (
                content.replace('```json', '')
                .replace('```', '')
                .replace('`', '')
                .strip()
            )
            
            # Remove any non-JSON text before or after
            start = content.find('{')
            end = content.rfind('}')
            
            if start >= 0 and end > start:
                json_str = content[start:end + 1]
                
                # Additional cleanup for common issues
                json_str = (
                    json_str.replace('\\n', '')
                    .replace('\n', '')
                    .replace('\\', '')
                    .replace('"{', '{')
                    .replace('}"', '}')
                    .strip()
                )
                
                # Fix common JSON formatting issues
                json_str = (
                    json_str.replace("'", '"')  # Replace single quotes with double quotes
                    .replace(': .', ': 0.')  # Fix leading decimal points
                    .replace(':.', ':0.')
                )
                
                self.logger.debug(f"Cleaned JSON string: {json_str}")
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error after cleaning: {str(e)}")
                    return None
            
            self.logger.error("No JSON object found in content")
            return None
                
        except Exception as e:
            self.logger.error(f"Error extracting JSON: {str(e)}")
            return None

    def _check_price_consistency(self, prices: List[float]) -> Optional[float]:
        """Check if prices are consistent."""
        if not prices:
            return None

        if len(prices) >= 2:
            mean = statistics.mean(prices)
            stdev = statistics.stdev(prices)
            cv = (stdev / mean) if mean > 0 else float('inf')

            if cv < 0.15:  # 15% variation threshold
                return statistics.median(prices)
            else:
                self.logger.warning(f"High price variation: {prices}")
                return statistics.median(prices)

        return prices[0] if prices else None

    async def get_price(self, item: str, unit: str) -> Dict:
        """Get validated wholesale price."""
        try:
            self.logger.info(f"Getting prices for {item} per {unit}")

            prices = await self._get_multiple_prices(item, unit)
            self.logger.info(f"Got {len(prices)} prices for {item}: {prices}")

            if prices:
                final_price = self._check_price_consistency(prices)
                if final_price is not None:
                    self.logger.info(f"Final price for {item}: ${final_price:.2f}/{unit}")
                    return {
                        'item': item,
                        'unit': unit,
                        'price': final_price,
                        'source': 'Wholesale Database'
                    }

            self.logger.warning(f"Using fallback price for {item}")
            return {
                'item': item,
                'unit': unit,
                'price': 0.25,
                'source': 'Fallback Price'
            }

        except Exception as e:
            self.logger.error(f"Error getting price for {item}: {str(e)}")
            return {
                'item': item,
                'unit': unit,
                'price': 0.25,
                'source': 'Error Fallback'
            }

    async def calculate_cost(self, items: List[Dict]) -> List[Dict]:
        """Calculate costs with validation."""
        results = []
        for item in items:
            try:
                price_info = await self.get_price(item['item'], item['unit'])
                total = Decimal(str(item['amount'])) * Decimal(str(price_info['price']))
                
                result = {
                    'item': item['item'],
                    'amount': item['amount'],
                    'unit': item['unit'],
                    'unit_price': price_info['price'],
                    'total_cost': float(total),
                    'source': price_info['source']
                }
                results.append(result)

            except Exception as e:
                self.logger.error(f"Error calculating cost for {item['item']}: {str(e)}")
                results.append({
                    'item': item['item'],
                    'amount': item['amount'],
                    'unit': item['unit'],
                    'unit_price': 0.25,
                    'total_cost': float(item['amount'] * 0.25),
                    'source': 'Error Fallback'
                })

        return results
