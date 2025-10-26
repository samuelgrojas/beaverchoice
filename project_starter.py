"""
Munder Difflin Paper Company - Multi-Agent System
=================================================

This solution implements a multi-agent system for managing quotes, 
inventory, and orders for a paper supply company.
"""

import time
import dotenv
import logging
import json
import re
import pandas as pd
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from smolagents import tool, ToolCallingAgent
from smolagents.models import LiteLLMModel

from database import *

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MunderDifflin")

# Load environment and initialize model
dotenv.load_dotenv()
model = LiteLLMModel(
    model_id="ollama/qwen2.5:7b",
    api_base="http://localhost:11434"
)

# ======= Helper Functions =======

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    # print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0

def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

def normalize_item_name(requested_name: str, available_items: List[str]) -> str:
    """
    Normalize item name to match available items.
    
    Args:
        requested_name (str): The name of the item to normalize.
        available_items (list): A list of available item names.
        
    Returns:
        str: The normalized item name.
    """
    requested_lower = requested_name.lower().strip()
    
    # Direct match
    for item in available_items:
        if item.lower() == requested_lower:
            return item
    
    # Keyword mappings
    keyword_mappings = {
        'glossy': ['Glossy paper'],
        'matte': ['Matte paper'],
        'cardstock': ['Cardstock'],
        'colored paper': ['Colored paper'],
        'construction paper': ['Construction paper'],
        'poster paper': ['Poster paper'],
        'poster board': ['Large poster paper (24x36 inches)', 'Poster paper'],
        'banner paper': ['Banner paper', 'Rolls of banner paper (36-inch width)'],
        'a4 paper': ['A4 paper'],
        'a4 white': ['A4 paper'],
        'a4 glossy': ['Glossy paper'],
        'a4 matte': ['Matte paper'],
        'a3 paper': ['A4 paper'],
        'a3 glossy': ['Glossy paper'],
        'a3 matte': ['Matte paper'],
        'a5 colored': ['Colored paper'],
        'printer paper': ['A4 paper'],
        'printing paper': ['A4 paper'],
        'white paper': ['A4 paper'],
        'copy paper': ['A4 paper'],
        'recycled': ['Recycled paper'],
        'kraft paper': ['Kraft paper'],
        'kraft envelope': ['Kraft paper'],
        'photo paper': ['Photo paper'],
        'heavyweight': ['Heavyweight paper', '100 lb cover stock'],
        'cover stock': ['100 lb cover stock'],
        'text paper': ['80 lb text paper'],
        'flyer': ['Flyers'],
        'poster': ['Large poster paper (24x36 inches)', 'Poster paper'],
        'ticket': ['Invitation cards'],
        'invitation': ['Invitation cards'],
        'envelope': ['Envelopes'],
        'napkin': ['Paper napkins'],
        'paper napkin': ['Paper napkins'],
        'table napkin': ['Paper napkins'],
        'paper plate': ['Paper plates'],
        'plate': ['Paper plates'],
        'paper cup': ['Paper cups'],
        'cup': ['Paper cups', 'Disposable cups'],
        'biodegradable cup': ['Disposable cups', 'Paper cups'],
        'biodegradable plate': ['Paper plates'],
        'streamer': ['Party streamers'],
        'washi tape': ['Decorative adhesive tape (washi tape)'],
        'presentation folder': ['Presentation folders'],
        'table cover': ['Table covers'],
    }
    
    for keyword, matches in keyword_mappings.items():
        if keyword in requested_lower:
            if matches is None:
                continue
            for match in matches:
                if match in available_items:
                    return match
    
    # Partial word matches
    requested_words = [w for w in requested_lower.split() if len(w) >= 4]
    for item in available_items:
        item_lower = item.lower()
        for word in requested_words:
            if word in item_lower:
                return item
    
    return requested_name

# ======= Tools =======

@tool
def list_available_products() -> List[str]:
    """
    List all available products.
    
    Returns:
        list: A list of available product names.
    """
    inventory_df = pd.read_sql("SELECT DISTINCT item_name FROM inventory", db_engine)
    return inventory_df["item_name"].tolist()

@tool
def get_item_price(item_name: str) -> Dict[str, any]:
    """
    Get the unit price and category of a specific item.
    
    Args:
        item_name (str): The name of the item to retrieve.
        
    Returns:
        dict: A dictionary containing the item name, unit price, and category.
    """
    inventory_df = pd.read_sql("SELECT * FROM inventory WHERE item_name = :name", 
                               db_engine, params={"name": item_name})
    if inventory_df.empty:
        return {"success": False, "message": f"Item '{item_name}' not found"}
    
    return {
        "success": True,
        "item_name": item_name,
        "unit_price": float(inventory_df["unit_price"].iloc[0]),
        "category": inventory_df["category"].iloc[0]
    }

@tool
def check_item_stock(item_name: str, request_date: str) -> Dict[str, any]:
    """
    Check the current stock level for a specific item.
    
    Args:
        item_name (str): The name of the item to check.
        request_date (str): The date for which to check stock.
        
    Returns:
        dict: A dictionary containing the item name and current stock level.
    """
    stock_df = get_stock_level(item_name, request_date)
    if stock_df.empty:
        return {"item_name": item_name, "current_stock": 0}
    return {
        "item_name": item_name,
        "current_stock": int(stock_df["current_stock"].iloc[0])
    }

@tool
def calculate_order_cost(items: Dict[str, int]) -> Dict[str, any]:
    """
    Calculate the total cost of an order.
    
    Args:
        items (dict): A dictionary of item names and quantities.
        
    Returns:
        dict: A dictionary containing the total cost, breakdown, and unavailable items.
    """
    available_items = list_available_products()
    total = 0.0
    breakdown = []
    unavailable = []
    
    for item_name, quantity in items.items():
        normalized = normalize_item_name(item_name, available_items)
        price_info = get_item_price(normalized)
        
        if not price_info.get("success"):
            unavailable.append(item_name)
            continue
        
        unit_price = price_info["unit_price"]
        line_total = unit_price * quantity
        total += line_total
        
        breakdown.append({
            "item_name": normalized,
            "quantity": quantity,
            "unit_price": unit_price,
            "line_total": line_total
        })
    
    return {
        "success": len(unavailable) == 0,
        "total_amount": total,
        "breakdown": breakdown,
        "unavailable_items": unavailable
    }

@tool
def order_stock_from_supplier(item_name: str, quantity: int, order_date: str) -> Dict[str, any]:
    """
    Order stock from a supplier.
    
    Args:
        item_name (str): The name of the item to order.
        quantity (int): The quantity of the item to order.
        order_date (str): The date of the order.
        
    Returns:
        dict: A dictionary containing the order result.
    """
    price_info = get_item_price(item_name)
    if not price_info.get("success"):
        return price_info
    
    unit_price = price_info["unit_price"]
    total_cost = unit_price * quantity
    
    current_cash = get_cash_balance(order_date)
    if current_cash < total_cost:
        return {
            "success": False,
            "message": f"Insufficient funds. Need ${total_cost:.2f}, have ${current_cash:.2f}"
        }
    
    delivery_date = get_supplier_delivery_date(order_date, quantity)
    
    transaction_id = create_transaction(
        item_name=item_name,
        transaction_type="stock_orders",
        quantity=quantity,
        price=total_cost,
        date=delivery_date
    )
    
    return {
        "success": True,
        "message": f"Ordered {quantity} units of {item_name}",
        "transaction_id": transaction_id,
        "total_cost": total_cost,
        "delivery_date": delivery_date
    }

@tool
def process_sale_transaction(items: Dict[str, int], sale_date: str) -> Dict[str, any]:
    """
    Process a sale transaction.
    
    Args:
        items (dict): A dictionary of item names and quantities.
        sale_date (str): The date of the sale.
        
    Returns:
        dict: A dictionary containing the sale result.
    """
    total_revenue = 0.0
    breakdown = []
    insufficient = []
    
    # Check stock first
    for item_name, quantity in items.items():
        stock_info = check_item_stock(item_name, sale_date)
        current_stock = stock_info["current_stock"]
        
        if current_stock < quantity:
            insufficient.append({
                "item_name": item_name,
                "requested": quantity,
                "available": current_stock
            })
    
    if insufficient:
        return {
            "success": False,
            "message": "Insufficient stock",
            "insufficient_stock": insufficient
        }
    
    # Process sale
    for item_name, quantity in items.items():
        price_info = get_item_price(item_name)
        unit_price = price_info["unit_price"]
        line_total = unit_price * quantity
        
        transaction_id = create_transaction(
            item_name=item_name,
            transaction_type="sales",
            quantity=quantity,
            price=line_total,
            date=sale_date
        )
        
        total_revenue += line_total
        breakdown.append({
            "item_name": item_name,
            "quantity": quantity,
            "unit_price": unit_price,
            "line_total": line_total,
            "transaction_id": transaction_id
        })
    
    return {
        "success": True,
        "message": "Sale completed",
        "total_revenue": total_revenue,
        "breakdown": breakdown
    }

# ======= Agents =======

class QuoteAgent(ToolCallingAgent):
    """Agent responsible for generating price quotes."""
    
    def __init__(self, model):
        super().__init__(
            tools=[list_available_products, get_item_price, calculate_order_cost],
            model=model,
            name="quote_specialist",
            description="Generates price quotes for customer orders."
        )

class InventoryAgent(ToolCallingAgent):
    """Agent responsible for managing inventory."""
    
    def __init__(self, model):
        super().__init__(
            tools=[check_item_stock, order_stock_from_supplier],
            model=model,
            name="inventory_manager",
            description="Manages inventory levels and orders from suppliers."
        )

class SalesAgent(ToolCallingAgent):
    """Agent responsible for processing sales."""
    
    def __init__(self, model):
        super().__init__(
            tools=[check_item_stock, process_sale_transaction],
            model=model,
            name="sales_processor",
            description="Processes customer sales transactions."
        )

# ======= Orchestrator =======

class MunderDifflinOrchestrator:
    """Coordinates the multi-agent paper company system."""
    
    def __init__(self, model):
        self.quote_agent = QuoteAgent(model)
        self.inventory_agent = InventoryAgent(model)
        self.sales_agent = SalesAgent(model)
    
    def extract_items_from_request(self, request: str) -> Dict[str, int]:
        """Extract items and quantities from customer request."""
        items = {}
        available_items = list_available_products()
        request_lower = request.lower()
        
        lines = re.split(r'[\n\r]|(?:,\s*and)|(?:\sand\s)', request_lower)
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue
            
            pattern = re.search(
                r'(\d+(?:,\d{3})*)\s+(?:sheets?\s+(?:of\s+)?|reams?\s+(?:of\s+)?|rolls?\s+(?:of\s+)?|packets?\s+(?:of\s+)?)?(.+?)(?:\s*\(|$)', 
                line
            )
            
            if pattern:
                quantity_str = pattern.group(1).replace(',', '')
                quantity = int(quantity_str)
                item_desc = pattern.group(2).strip()
                
                item_desc = re.sub(r'\(.*?\)', '', item_desc).strip()
                item_desc = re.sub(r'[,.]$', '', item_desc).strip()
                
                if 'ream' in line:
                    quantity = quantity * 500
                
                normalized = normalize_item_name(item_desc, available_items)
                
                if normalized in available_items:
                    if normalized in items:
                        items[normalized] += quantity
                    else:
                        items[normalized] = quantity
                    logger.info(f"Extracted: {item_desc} → {normalized} (qty: {quantity})")
        
        return items
    
    def extract_quote_details(self, response: str) -> Dict[str, any]:
        """Extract quote details from agent response."""
        details = {}
        
        # Extract total amount
        total_match = re.search(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', response)
        if total_match:
            details["total"] = float(total_match.group(1).replace(',', ''))
        
        return details
    
    def process_order(self, customer_request: str, request_date: str) -> str:
        """
        Process a customer order from request through completion.
        
        Args:
            customer_request: Natural language order request
            request_date: Date of the request
            
        Returns:
            Response to customer with order details
        """
        logger.info(f"Processing order for date: {request_date}")
        
        # Step 1: Extract items
        items_dict = self.extract_items_from_request(customer_request)
        
        if not items_dict:
            return "I couldn't identify the items you need. Please specify item names and quantities."
        
        logger.info(f"Extracted items: {items_dict}")
        
        # Step 2: Generate quote
        quote_response = self.quote_agent.run(
            f"""Customer wants: {json.dumps(items_dict)}
            
            Use calculate_order_cost to get pricing.
            Format a professional quote with itemized breakdown and total.
            """
        )
        
        logger.info(f"Quote: {quote_response[:150]}")
        
        # Step 3: Check and manage inventory
        inventory_response = self.inventory_agent.run(
            f"""Check stock for items: {json.dumps(items_dict)}
            Date: {request_date}
            
            For each item, use check_item_stock.
            If any item stock < quantity needed:
            - Calculate shortage
            - Use order_stock_from_supplier to order 2x the shortage
            
            Report status of each item.
            """
        )
        
        logger.info(f"Inventory: {inventory_response[:150]}")
        
        # Check if inventory mentioned any issues
        inventory_issue = any(term in inventory_response.lower() for term in 
                            ["insufficient", "not enough", "shortage", "low stock"])
        
        if inventory_issue:
            # Don't process sale yet
            return f"{quote_response}\n\n{inventory_response}\n\nWe're ordering additional stock to fulfill your order."
        
        # Step 4: Process the sale
        sales_response = self.sales_agent.run(
            f"""Process sale for: {json.dumps(items_dict)}
            Date: {request_date}
            
            First check stock with check_item_stock for all items.
            If all items available, use process_sale_transaction.
            
            Calculate delivery date as {request_date} + 14 days.
            Return confirmation with order details and delivery date.
            """
        )
        
        logger.info(f"Sales: {sales_response[:150]}")
        
        # Check if sale was successful
        sale_failed = any(term in sales_response.lower() for term in 
                         ["insufficient", "failed", "error", "cannot"])
        
        if sale_failed:
            return f"{quote_response}\n\n{sales_response}"
        
        # Calculate delivery date
        delivery_date = (datetime.fromisoformat(request_date) + timedelta(days=14)).strftime("%Y-%m-%d")
        
        # Build final response
        final_response = f"{quote_response}\n\nORDER CONFIRMED!\n"
        final_response += f"Order Date: {request_date}\n"
        final_response += f"Expected Delivery: {delivery_date}\n\n"
        final_response += "Thank you for your business!"
        
        return final_response

# ======= Test Scenarios =======

def run_test_scenarios():
    """Run test scenarios from CSV file."""
    logger.info("Initializing Database...")
    init_database(db_engine)
    
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return
    
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]
    
    orchestrator = MunderDifflinOrchestrator(model)
    
    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")
        
        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Date: {request_date}")
        print(f"Cash: ${current_cash:.2f} | Inventory: ${current_inventory:.2f}")
        
        try:
            response = orchestrator.process_order(row['request'], request_date)
        except Exception as e:
            response = f"Error: {str(e)}"
            logger.error(f"Error processing request: {e}", exc_info=True)
        
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]
        
        print(f"Response: {response[:200]}...")
        print(f"Updated Cash: ${current_cash:.2f} | Inventory: ${current_inventory:.2f}")
        
        results.append({
            "request_id": idx + 1,
            "request_date": request_date,
            "cash_balance": current_cash,
            "inventory_value": current_inventory,
            "response": response,
        })
        
        time.sleep(1)
    
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL REPORT =====")
    print(f"Cash: ${final_report['cash_balance']:.2f}")
    print(f"Inventory: ${final_report['inventory_value']:.2f}")
    
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results

if __name__ == "__main__":
    run_test_scenarios()