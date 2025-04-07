# function_app.py
import os
import json
import logging
import tempfile
import mimetypes
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional

import azure.functions as func
import azure.durable_functions as df
from azure.storage.blob.aio import BlobServiceClient
from azure.cosmos.aio import CosmosClient
from azure.servicebus.aio import ServiceBusClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.search.documents import SearchClient
from azure.servicebus import ServiceBusMessage
from openai import OpenAI

from shared.Agent_tools import AgentIngredientMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('recipe_processor')

# Initialize function app
myapp = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Initialize blob client
blob_service_client = BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])

async def get_or_create_container(user_id: str):
    """Get or create a container for the user."""
    container_name = ''.join(c.lower() for c in f"user{user_id}recipes" if c.isalnum() or c == '-')
    logging.info(f"Attempting to create/access container: {container_name}")
    container_client = blob_service_client.get_container_client(container_name)
    
    try:
        # Try to create the container
        await container_client.create_container()
        logging.info(f"Created new container: {container_name}")
    except Exception as e:
        if "ContainerAlreadyExists" in str(e):
            logging.info(f"Container already exists: {container_name}")
        else:
            logging.error(f"Error creating container: {str(e)}")
            raise
    
    return container_client, container_name

def is_valid_file_type(filename: str) -> bool:
    """Check if file type is allowed."""
    allowed_extensions = {'.pdf', '.xlsx', '.xls', '.csv', '.jpg', '.jpeg', '.png'}
    return os.path.splitext(filename)[1].lower() in allowed_extensions

@myapp.route(route="processrecipes/{user_id}/{index_name}", methods=["POST"])
@myapp.durable_client_input(client_name="client")
async def http_trigger(req: func.HttpRequest, client) -> func.HttpResponse:
    """HTTP trigger for recipe processing."""
    try:
        user_id = req.route_params.get('user_id')
        index_name = req.route_params.get('index_name')

        if not user_id:
            return func.HttpResponse(
                json.dumps({"error": "User ID is required"}),
                mimetype="application/json",
                status_code=400
            )
        if not index_name:
            return func.HttpResponse(
                json.dumps({"error": "Index name is required"}),
                mimetype="application/json",
                status_code=400
            )

        # Get or create the container
        container_client, container_name = await get_or_create_container(user_id)

        # Handle file uploads
        blob_references = []
        max_file_size = 50 * 1024 * 1024  # 50MB limit
        
        for file_name in req.files:
            file = req.files[file_name]
            
            if not is_valid_file_type(file.filename):
                return func.HttpResponse(
                    json.dumps({"error": f"Invalid file type: {file.filename}"}),
                    mimetype="application/json",
                    status_code=400
                )

            file_content = file.read()
            if len(file_content) > max_file_size:
                return func.HttpResponse(
                    json.dumps({"error": f"File too large: {file.filename}"}),
                    mimetype="application/json",
                    status_code=400
                )
            file.seek(0)

            # Use sanitized filename for blob
            safe_index_name = ''.join(c for c in index_name if c.isalnum() or c in '._-')
            safe_filename = ''.join(c for c in file.filename if c.isalnum() or c in '._-')
            blob_name = f"{safe_index_name}/{safe_filename}"
            
            # Create blob client using the existing container
            blob_client = container_client.get_blob_client(blob_name)
            
            # Upload the file
            await blob_client.upload_blob(file.stream, overwrite=True)
            blob_references.append({
                "blob_name": blob_name,
                "container_name": container_name
            })

        if not blob_references:
            return func.HttpResponse(
                json.dumps({"error": "No files uploaded"}),
                mimetype="application/json",
                status_code=400
            )

        # Start orchestration
        instance_id = await client.start_new(
            "recipe_orchestrator",
            None,
            {
                "user_id": user_id,
                "index_name": index_name,
                "blobs": blob_references
            }
        )
        
        return client.create_check_status_response(req, instance_id)

    except Exception as e:
        logging.error(f"Error in http_trigger: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Internal server error", "details": str(e)}),
            mimetype="application/json",
            status_code=500
        )

@myapp.orchestration_trigger(context_name="context")
def recipe_orchestrator(context: df.DurableOrchestrationContext):
    """Orchestrator function for recipe processing."""
    try:
        input_data = context.get_input()
        if isinstance(input_data, str):
            input_data = json.loads(input_data)
            
        user_id = input_data.get("user_id")
        index_name = input_data.get("index_name")
        blobs = input_data.get("blobs", [])

        if not user_id or not blobs or not index_name:
            return {
                "status": "failed",
                "message": "Invalid input data",
                "recipe_count": 0
            }

        # Process files in parallel
        tasks = []
        for blob in blobs:
            task = context.call_activity("process_file_agent", {
                "blob": blob,
                "index_name": index_name,
                "user_id": user_id
            })
            tasks.append(task)

        results = yield context.task_all(tasks)
        valid_results = [r for result in results for r in result if r and isinstance(r, dict)]
        
        if not valid_results:
            return {
                "status": "completed",
                "message": "No valid recipes found",
                "recipe_count": 0
            }

        # Store results
        store_result = yield context.call_activity("store_recipes_activity", {
            "user_id": user_id,
            "index_name": index_name,
            "recipes": valid_results
        })

        return {
            "status": "completed",
            "message": f"Processed {len(valid_results)} recipes",
            "recipe_count": len(valid_results),
            "total_user_recipes": store_result.get("total_user_recipes", 0)
        }

    except Exception as e:
        logging.error(f"Error in recipe_orchestrator: {str(e)}")
        return {"status": "failed", "error": str(e)}

@myapp.activity_trigger(input_name="taskinput")
async def process_file_agent(taskinput):
    """Activity function for processing individual files using the agent approach."""
    temp_dir = None
    try:
        blob_info = taskinput.get("blob")
        index_name = taskinput.get("index_name")
        user_id = taskinput.get("user_id")

        if not blob_info or not index_name or not user_id:
            return []

        # Initialize clients
        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        form_client = DocumentAnalysisClient(
            endpoint=os.environ["AZURE_FORM_RECOGNIZER_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_FORM_RECOGNIZER_KEY"])
        )
        
        # Initialize blob service client
        blob_service_client = BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])
        container_client = blob_service_client.get_container_client(blob_info["container_name"])
        blob_client = container_client.get_blob_client(blob_info["blob_name"])

        # Download and process the blob
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, os.path.basename(blob_info["blob_name"]))
        
        download_stream = await blob_client.download_blob()
        with open(temp_path, "wb") as temp_file:
            async for chunk in download_stream.chunks():
                temp_file.write(chunk)

        # Initialize the agent
        ingredient_agent = AgentIngredientMatcher(openai_client, user_id)
        
        # Extract recipe content
        recipes_data = await extract_recipes_with_agent(temp_path, form_client, openai_client)
        
        if not recipes_data or not recipes_data.get("recipes"):
            return []
            
        processed_recipes = []
        for recipe in recipes_data["recipes"]:
            # Process recipe with agent
            result = await process_recipe_with_agent(recipe, ingredient_agent, openai_client)
            if result:
                processed_recipes.append(result)
                
        return processed_recipes

    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return []
    finally:
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)

@myapp.activity_trigger(input_name="storeinput")
async def store_recipes_activity(storeinput: Dict[str, Any]) -> Dict:
    """Activity function for storing processed recipes."""
    try:
        user_id = storeinput.get("user_id")
        recipes = storeinput.get("recipes", [])
        index_name = storeinput.get("index_name")

        if not user_id or not index_name:
            raise ValueError("User ID and index name are required")

        if not recipes:
            return {
                "status": "completed",
                "message": "No recipes to store",
                "stored_count": 0,
                "total_user_recipes": 0
            }

        async with CosmosClient(
            url=os.environ["COSMOS_ENDPOINT"],
            credential=os.environ["COSMOS_KEY"]
        ) as cosmos_client:
            database = cosmos_client.get_database_client("InvoicesDB")
            container = database.get_container_client("Recipes")

            # Get current user document or create new one
            try:
                user_doc = await container.read_item(
                    item=user_id,
                    partition_key=user_id
                )
            except:
                user_doc = {
                    "id": user_id,
                    "type": "user",
                    "recipe_count": 0,
                    "recipes": {}
                }

            # Update recipe count and add new recipes
            current_count = user_doc.get("recipe_count", 0)
            
            # Organize recipes by index
            if index_name not in user_doc["recipes"]:
                user_doc["recipes"][index_name] = []

            # Add new recipes with sequential numbering
            for i, recipe in enumerate(recipes, start=current_count + 1):
                recipe_entry = {
                    "id": f"{user_id}_{index_name}_{i}",
                    "sequence_number": i,
                    "name": recipe["recipe_name"],
                    "created_at": datetime.utcnow().isoformat(),
                    "data": recipe
                }
                user_doc["recipes"][index_name].append(recipe_entry)

            # Update total recipe count and timestamp
            user_doc["recipe_count"] = current_count + len(recipes)
            user_doc["last_updated"] = datetime.utcnow().isoformat()

            # Save updated user document
            await container.upsert_item(user_doc)

            # Send notification to Service Bus
            async with ServiceBusClient.from_connection_string(
                os.environ["ServiceBusConnection"]
            ) as servicebus_client:
                sender = servicebus_client.get_queue_sender("recipe-updates")
                message = {
                    "user_id": user_id,
                    "index_name": index_name,
                    "new_recipes": len(recipes),
                    "total_recipes": user_doc["recipe_count"],
                    "status": "completed"
                }
                await sender.send_messages([
                    ServiceBusMessage(json.dumps(message))
                ])

            return {
                "status": "completed",
                "message": f"Successfully stored {len(recipes)} recipes",
                "stored_count": len(recipes),
                "total_user_recipes": user_doc["recipe_count"]
            }

    except Exception as e:
        logging.error(f"Error storing recipes: {str(e)}")
        raise

async def extract_recipes_with_agent(file_path: str, form_client, openai_client) -> Dict:
    """Extract recipes using Form Recognizer and OpenAI."""
    try:
        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.pdf', '.jpg', '.jpeg', '.png']:
            # Extract text from PDF or image
            with open(file_path, "rb") as doc:
                result = form_client.begin_analyze_document("prebuilt-layout", doc).result()
                content = {
                    "tables": [[cell.content.strip() for cell in table.cells] for table in result.tables],
                    "text": [p.content.strip() for p in result.paragraphs]
                }
        elif file_ext in ['.xlsx', '.xls', '.csv']:
            # For Excel files
            import pandas as pd
            df = pd.read_excel(file_path) if file_ext in ['.xlsx', '.xls'] else pd.read_csv(file_path)
            content = {
                "tables": [df.values.tolist()],
                "text": df.to_string().split('\n')
            }
        else:
            return {"recipes": []}
            
        # Use OpenAI to extract structured recipe data
        system_prompt = """You are a precise recipe extraction specialist. Extract standardized recipe information from the source.
        
        OUTPUT STRUCTURE:
        {
            "recipes": [
                {
                    "name": "Complete Recipe Name",
                    "servings": number,
                    "items_per_serving": number,
                    "ingredients": [
                        {
                            "item": "ingredient base name",
                            "amount": number,
                            "unit": "standardized unit"
                        }
                    ],
                    "topping": "complete topping instructions",
                    "Type": "Recipe"
                }
            ]
        }
        
        STANDARDIZATION RULES:
        1. Units: Use standard units only (g, kg, oz, lb, ml, l, tbsp, tsp, cup, etc.)
        2. Convert all text numbers to numeric values
        3. Extract all ingredients with exact measurements
        4. Identify the number of servings
        
        Return ONLY the JSON with no additional text or explanations."""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(content)}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Error extracting recipes: {str(e)}")
        return {"recipes": []}

async def process_recipe_with_agent(recipe: Dict, ingredient_agent: AgentIngredientMatcher, openai_client: OpenAI) -> Dict:
    """Process a single recipe using the ingredient agent."""
    try:
        recipe_result = {
            "recipe_name": recipe["name"],
            "servings": recipe.get("servings", 1),
            "items_per_serving": recipe.get("items_per_serving"),
            "serving_size": recipe.get("serving_size"),
            "total_yield": recipe.get("total_yield", None),
            "topping": recipe.get("topping", ""),
            "Type": recipe.get("Type", "Recipe"),
            "location": recipe.get("location", "NA"),
            "ingredients": [],
            "total_cost": 0.0,
            "status":"Active",
            "cost_per_serving": 0.0
        }
        
        # Process each ingredient with the agent
        for ingredient in recipe["ingredients"]:
            # Get the best match using the agent
            match = await ingredient_agent.get_best_match(ingredient)
            
            if match:
                # If we have a match, calculate costs
                ingredient_cost = await calculate_ingredient_cost(ingredient, match, recipe_result["servings"], openai_client)
                recipe_result["ingredients"].append(ingredient_cost)
                recipe_result["total_cost"] += ingredient_cost["total_cost"]
            else:
                # If no match, try web search for pricing
                price_info = await search_ingredient_price(ingredient["item"], ingredient["unit"], openai_client)
                if price_info:
                    # Add item to inventory for future use
                    await ingredient_agent.add_to_cosmos(price_info, ingredient["item"])
                    
                    # Calculate costs
                    ingredient_cost = {
                        "ingredient": ingredient["item"],
                        "recipe_amount": f"{ingredient['amount']} {ingredient['unit']}",
                        "inventory_item": ingredient["item"],
                        "inventory_unit": price_info["unit"],
                        "converted_amount": float(ingredient["amount"]),  # Simple 1:1 conversion
                        "unit_cost": float(price_info["price"]),
                        "total_cost": float(price_info["price"]) * float(ingredient["amount"]),
                        "per_serving": {
                            "amount": float(ingredient["amount"]) / recipe_result["servings"],
                            "unit": ingredient["unit"],
                            "converted_amount": float(ingredient["amount"]) / recipe_result["servings"],
                            "cost": (float(price_info["price"]) * float(ingredient["amount"])) / recipe_result["servings"]
                        },
                        "is_retail_estimate": True
                    }
                    recipe_result["ingredients"].append(ingredient_cost)
                    recipe_result["total_cost"] += ingredient_cost["total_cost"]
        
        # Calculate cost per serving
        if recipe_result["servings"] > 0:
            recipe_result["cost_per_serving"] = recipe_result["total_cost"] / recipe_result["servings"]
            
        return recipe_result
    except Exception as e:
        logging.error(f"Error processing recipe {recipe.get('name', 'unknown')}: {str(e)}")
        return None

async def calculate_ingredient_cost(ingredient: Dict, match: Dict, servings: int, openai_client: OpenAI) -> Dict:
    """Calculate the cost of an ingredient with unit conversion if needed."""
    try:
        recipe_amount = float(ingredient["amount"])
        recipe_unit = ingredient["unit"]
        inventory_unit = match["unit"]
        
        # If units don't match, use OpenAI to convert
        converted_amount = recipe_amount
        if recipe_unit.lower() != inventory_unit.lower():
            converted_amount = await convert_units(
                amount=recipe_amount,
                from_unit=recipe_unit,
                to_unit=inventory_unit,
                ingredient=ingredient["item"],
                openai_client=openai_client
            )
        
        unit_cost = float(match["cost_per_unit"])
        total_cost = converted_amount * unit_cost
        
        per_serving_amount = recipe_amount / servings
        per_serving_converted = converted_amount / servings
        per_serving_cost = total_cost / servings
        
        return {
            "ingredient": ingredient["item"],
            "recipe_amount": f"{recipe_amount} {recipe_unit}",
            "inventory_item": match["inventory_item"],
            "inventory_unit": inventory_unit,
            "converted_amount": float(converted_amount),
            "unit_cost": float(unit_cost),
            "total_cost": float(total_cost),
            "per_serving": {
                "amount": float(per_serving_amount),
                "unit": recipe_unit,
                "converted_amount": float(per_serving_converted),
                "cost": float(per_serving_cost)
            },
            "is_retail_estimate": match.get("is_retail_estimate", False)
        }
    except Exception as e:
        logging.error(f"Error calculating ingredient cost: {str(e)}")
        # Return a default structure with zeros
        return {
            "ingredient": ingredient["item"],
            "recipe_amount": f"{ingredient['amount']} {ingredient['unit']}",
            "inventory_item": match.get("inventory_item", "Unknown"),
            "inventory_unit": match.get("unit", ingredient["unit"]),
            "converted_amount": 0,
            "unit_cost": 0,
            "total_cost": 0,
            "per_serving": {
                "amount": 0,
                "unit": ingredient["unit"],
                "converted_amount": 0,
                "cost": 0
            },
            "is_retail_estimate": match.get("is_retail_estimate", False)
        }

async def convert_units(amount: float, from_unit: str, to_unit: str, ingredient: str, openai_client: OpenAI) -> float:
    """Convert between different units using OpenAI."""
    try:
        # Common conversions dictionary
        common_conversions = {
            ("tbsp", "oz"): 0.5,      # 1 tbsp = 0.5 oz
            ("tsp", "oz"): 0.167,     # 1 tsp = 0.167 oz
            ("cup", "oz"): 8.0,       # 1 cup = 8 oz
            ("oz", "g"): 28.35,       # 1 oz = 28.35 g
            ("g", "kg"): 0.001,       # 1 g = 0.001 kg
            ("lb", "oz"): 16.0,       # 1 lb = 16 oz
            ("ml", "l"): 0.001,       # 1 ml = 0.001 l
            ("l", "ml"): 1000,        # 1 l = 1000 ml
            ("ml", "oz"): 0.033814,   # 1 ml = 0.033814 oz
            ("oz", "ml"): 29.5735     # 1 oz = 29.5735 ml
        }
        
        # Standardize units
        from_unit_std = from_unit.lower().strip()
        to_unit_std = to_unit.lower().strip()
        
        # Check if we have a predefined conversion
        if (from_unit_std, to_unit_std) in common_conversions:
            return amount * common_conversions[(from_unit_std, to_unit_std)]
        elif (to_unit_std, from_unit_std) in common_conversions:
            return amount / common_conversions[(to_unit_std, from_unit_std)]
            
        # If not found in common conversions, use OpenAI for more complex conversions
        system_prompt = """You are a culinary unit conversion expert. Convert the given ingredient amount from one unit to another.
        
        Return ONLY the converted amount as a number with no text, units, or explanation.
        For example, if asked to convert 2 tbsp of flour to grams, just return "15" not "15g" or "15 grams".
        
        Be precise with ingredient-specific densities and characteristics."""
        
        user_prompt = f"Convert {amount} {from_unit} of {ingredient} to {to_unit}. Return only the number."
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        # Extract the converted value
        converted_value = response.choices[0].message.content.strip()
        try:
            return float(converted_value)
        except ValueError:
            logging.error(f"Could not convert response to float: {converted_value}")
            # Return a reasonable fallback
            return amount
            
    except Exception as e:
        logging.error(f"Error converting units: {str(e)}")
        # Return original amount as fallback
        return amount


async def search_ingredient_price(ingredient: str, unit: str, openai_client: OpenAI) -> Optional[Dict]:
    """Search for ingredient price using web search with OpenAI."""
    try:
        # Format search query for food ingredient pricing
        search_query = f"wholesale price of {ingredient} per {unit} for food service"
        
        # Use OpenAI's web search capability
        search_response = openai_client.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={},
            messages=[
                {
                    "role": "system", 
                    "content": """You are a wholesale food pricing database extracting prices from search results.
                    Find the most accurate current wholesale price for the ingredient in the specified unit.
                    Consider food service industry prices, not retail prices."""
                },
                {
                    "role": "user",
                    "content": f"What is the current wholesale price of {ingredient} per {unit} for food service businesses?"
                }
            ]
        )
        
        # Extract the search results and analysis
        search_result = search_response.choices[0].message.content
        
        # Use a follow-up completion to extract structured data
        extraction_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """Extract wholesale food pricing information from the search results.
                    Return in this exact JSON format:
                    {
                        "item": "ingredient name",
                        "unit": "standardized unit",
                        "price": number,
                        "source": "source name"
                    }
                    
                    The price should be a realistic wholesale price per unit for food service businesses.
                    If multiple prices are found, use the most reliable/recent source.
                    If you cannot determine a specific price, provide a reasonable estimate based on similar ingredients."""
                },
                {
                    "role": "user", 
                    "content": f"Based on these search results, extract the wholesale price for 1 {unit} of {ingredient}:\n\n{search_result}"
                }
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        price_data = json.loads(extraction_response.choices[0].message.content)
        
        # Validate the response has the expected fields
        if all(key in price_data for key in ["item", "unit", "price", "source"]):
            logging.info(f"Found price for {ingredient}: ${price_data['price']} per {price_data['unit']} (Source: {price_data['source']})")
            return price_data
        else:
            logging.error(f"Invalid price data format: {price_data}")
            
            # Create fallback price structure
            return {
                "item": ingredient,
                "unit": unit,
                "price": estimate_fallback_price(ingredient, unit),
                "source": "Estimation"
            }
            
    except Exception as e:
        logging.error(f"Error searching for ingredient price: {str(e)}")
        return {
            "item": ingredient,
            "unit": unit,
            "price": estimate_fallback_price(ingredient, unit),
            "source": "Fallback Estimation"
        }

def estimate_fallback_price(ingredient: str, unit: str) -> float:
    """Provide a reasonable fallback price estimate for an ingredient."""
    # Basic categories with typical price ranges per common unit
    categories = {
        # Format: category: (base_price, common_unit)
        "meat": (5.00, "lb"),
        "seafood": (8.00, "lb"),
        "dairy": (2.50, "lb"),
        "produce": (1.50, "lb"),
        "grains": (1.00, "lb"),
        "spices": (0.75, "oz"),
        "herbs": (0.50, "oz"),
        "oils": (0.30, "oz"),
        "condiments": (0.25, "oz"),
        "default": (1.00, "unit")
    }
    
    # Simple keyword matching to determine category
    ingredient_lower = ingredient.lower()
    for category, (base_price, base_unit) in categories.items():
        if category in ingredient_lower:
            if unit.lower() == base_unit.lower():
                return base_price
            else:
                # Basic unit conversion for common cases
                if base_unit == "lb" and unit.lower() == "oz":
                    return base_price / 16.0
                elif base_unit == "oz" and unit.lower() == "lb":
                    return base_price * 16.0
    
    # Default fallback if no category match
    return 1.00