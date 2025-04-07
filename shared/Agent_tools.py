import os
import json
import logging
import time
from decimal import Decimal
from typing import Dict, List, Optional, Any
from datetime import datetime
from fuzzywuzzy import fuzz
from openai import OpenAI
from azure.cosmos import CosmosClient

class AgentIngredientMatcher:
    """Intelligent ingredient matching using OpenAI Assistant API."""

    def __init__(self, openai_client, user_id: str):
        self.openai_client = openai_client
        self.user_id = user_id
        self.price_scraper = None  # We'll still use this for retail pricing if needed
        
        # Initialize CosmosDB client
        self.cosmos_container = CosmosClient(
            os.getenv("COSMOS_ENDPOINT"),
            credential=os.getenv("COSMOS_KEY")
        ).get_database_client(
            os.getenv("COSMOS_DATABASE_ID")
        ).get_container_client(
            os.getenv("COSMOS_CONTAINER_ID")
        )
        
        # Create the assistant once and reuse it
        self.assistant_id = self._create_ingredient_assistant()
        
    def _create_ingredient_assistant(self) -> str:
        """Create an OpenAI Assistant with inventory matching capabilities."""
        # Define the tool for inventory search
        inventory_search_tool = {
            "type": "function",
            "function": {
                "name": "search_inventory",
                "description": "Search for matching ingredients in the user's inventory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ingredient_name": {
                            "type": "string",
                            "description": "The name of the ingredient to search for"
                        },
                        "amount": {
                            "type": "number",
                            "description": "The amount of the ingredient needed"
                        },
                        "unit": {
                            "type": "string",
                            "description": "The unit of measurement for the ingredient"
                        }
                    },
                    "required": ["ingredient_name"]
                }
            }
        }
        
        # Create the assistant with the tool
        assistant = self.openai_client.beta.assistants.create(
            name="Ingredient Matcher",
            instructions="""You are an expert at matching recipe ingredients to inventory items. 
            Your job is to find the best matching inventory item for each recipe ingredient.
            
            When evaluating matches, consider:
            1. Ingredient name similarity (most important)
            2. Unit compatibility (can the units be reasonably converted?)
            3. Cost reasonableness
            
            For name matching, consider:
            - Exact matches or plural forms (e.g. 'tomato' vs 'tomatoes') are perfect matches
            - Same ingredient different form (e.g. 'garlic' vs 'garlic powder') are good matches
            - Common substitutes (e.g. 'vegetable oil' vs 'canola oil') are acceptable matches
            
            Always select the most appropriate inventory item that aligns with the recipe ingredient.
            If no suitable match exists, indicate that clearly.
            """,
            model="gpt-4o",
            tools=[inventory_search_tool]
        )
        
        return assistant.id
    
    async def _handle_inventory_search(self, ingredient_name: str, amount: float = None, unit: str = None) -> Dict:
        """
        Handle the inventory search tool call by querying the user's inventory in CosmosDB.
        Returns potential matches with similarity scores.
        """
        try:
            logging.info(f"Searching for ingredient: {ingredient_name} for user: {self.user_id}")
            
            # Query the user document
            query = "SELECT * FROM c WHERE c.id = @user_id"
            params = [{"name": "@user_id", "value": self.user_id}]
            
            results = list(self.cosmos_container.query_items(
                query=query,
                parameters=params,
                enable_cross_partition_query=True
            ))
            
            if not results:
                logging.warning(f"No document found for user: {self.user_id}")
                return {
                    "success": False,
                    "error": "User inventory not found",
                    "matches": []
                }
                
            user_doc = results[0]
            if 'items' not in user_doc:
                logging.warning(f"No items array in user document: {self.user_id}")
                return {
                    "success": False,
                    "error": "User has no inventory items",
                    "matches": []
                }
                
            # Calculate fuzzy match scores for all inventory items
            matches = []
            for item in user_doc['items']:
                try:
                    if item.get('Active', 'No').lower() != 'yes':
                        continue
                        
                    inventory_name = item.get('Inventory Item Name')
                    if not inventory_name:
                        continue
                    
                    # Calculate fuzzy match score
                    score = fuzz.ratio(ingredient_name.lower(), inventory_name.lower())
                    
                    # Include items with a reasonable score (50+ is a starting point)
                    if score >= 50:
                        matches.append({
                            'inventory_item': inventory_name,
                            'cost_per_unit': float(item['Cost of a Unit']),
                            'measured_in': item['Measured In'],
                            'supplier_name': item['Supplier Name'],
                            'match_score': score
                        })
                except Exception as item_error:
                    logging.error(f"Error processing item: {str(item_error)}")
                    continue
            
            # Sort matches by score (highest first)
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Return top matches (limit to 5)
            return {
                "success": True,
                "matches": matches[:5]
            }
                
        except Exception as e:
            logging.error(f"Inventory search error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "matches": []
            }
    
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
            self.cosmos_container.replace_item(
                item=user_doc['id'],
                body=user_doc
            )
            
            print(f"Added new inventory item: {ingredient_name}")
            
        except Exception as e:
            print(f"Error adding to CosmosDB: {str(e)}")

    async def get_best_match(self, ingredient: Dict) -> Optional[Dict]:
        """
        Use the OpenAI Assistant to find the best match for an ingredient.
        This creates a thread, sends the ingredient info, and handles tool calls.
        """
        try:
            # Create a thread for this conversation
            thread = self.openai_client.beta.threads.create()
            
            # Format the request message
            message_content = f"""
            I need to find the best inventory match for this recipe ingredient:
            
            Ingredient: {ingredient['item']}
            Amount: {ingredient['amount']}
            Unit: {ingredient['unit']}
            
            Please search our inventory and recommend the best match.
            """
            
            # Add the message to the thread
            self.openai_client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=message_content
            )
            
            # Run the assistant to get a response
            run = self.openai_client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            
            # Poll for completion or tool calls
            while run.status != "completed":
                run = self.openai_client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                
                if run.status == "requires_action":
                    # Handle tool calls
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls
                    tool_outputs = []
                    
                    for tool_call in tool_calls:
                        if tool_call.function.name == "search_inventory":
                            # Parse the function arguments
                            args = json.loads(tool_call.function.arguments)
                            
                            # Execute the inventory search
                            search_result = await self._handle_inventory_search(
                                args.get("ingredient_name", ingredient['item']),
                                args.get("amount", ingredient['amount']),
                                args.get("unit", ingredient['unit'])
                            )
                            
                            # Add the result to tool outputs
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": json.dumps(search_result)
                            })
                    
                    # Submit tool outputs back to the assistant
                    run = self.openai_client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
                
                # Avoid excessive polling
                if run.status not in ["completed", "requires_action"]:
                    time.sleep(1)
            
            # Get the assistant's response
            messages = self.openai_client.beta.threads.messages.list(
                thread_id=thread.id
            )
            
            # The most recent message from the assistant should contain the recommendation
            for message in messages.data:
                if message.role == "assistant":
                    # Parse the assistant's response to extract the recommended match
                    response_text = message.content[0].text.value
                    
                    # Use another GPT call to extract structured data from the response
                    extraction_response = self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "system", 
                                "content": """Extract the best matching inventory item from the assistant's response.
                                Return a JSON object with these fields:
                                - inventory_item: The name of the matched item
                                - cost_per_unit: The cost per unit as a number
                                - unit: The unit of measurement
                                - supplier: The supplier name
                                - location: The location of the inventory item (if mentioned) if not NA
                                - similarity: A number between 0 and 1 representing match quality
                                
                                If no match was found, return {"no_match": true}
                                """
                            },
                            {"role": "user", "content": response_text}
                        ],
                        response_format={"type": "json_object"}
                    )
                    
                    match_data = json.loads(extraction_response.choices[0].message.content)
                    
                    # Check if no match was found
                    if match_data.get("no_match", False):
                        if self.price_scraper:
                            # Try to get a retail price estimate
                            price_data = await self.price_scraper.get_price(
                                item=ingredient['item'],
                                unit=ingredient['unit']
                            )
                            if price_data:
                                await self.add_to_cosmos(price_data, ingredient['item'])
                                return {
                                    'inventory_item': ingredient['item'],
                                    'cost_per_unit': Decimal(str(price_data['price'])),
                                    'unit': price_data['unit'],
                                    'supplier': f"Retail ({price_data['source'].capitalize()})",
                                    'is_retail_estimate': True
                                }
                        return None
                    
                    # Convert cost to Decimal for precision
                    if 'cost_per_unit' in match_data:
                        match_data['cost_per_unit'] = Decimal(str(match_data['cost_per_unit']))
                    
                    return match_data
            
            return None
            
        except Exception as e:
            print(f"Error in get_best_match for {ingredient['item']}: {str(e)}")
            return None