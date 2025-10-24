import unittest
from project_starter import (init_database, db_engine, check_inventory, reorder_item, get_delivery_estimate, generate_quote, get_quote_history, fulfill_order)

init_database(db_engine)

class TestTools(unittest.TestCase):
    def test_check_inventory(self):
        # Prueba para la función check_inventory
        item_name = "A4 paper"
        date = "2025-04-01"
        resultado_obtenido = check_inventory(item_name, date)
        self.assertIsInstance(resultado_obtenido, int)

    def test_reorder_item(self):
        # Prueba para la función reorder_item
        item_name = "A4 paper"
        quantity = 300
        date = "2025-04-01"
        resultado_obtenido = reorder_item(item_name, quantity, date)
        self.assertIsInstance(resultado_obtenido, int)

    def test_get_delivery_estimate(self):
        # Prueba para la función get_delivery_estimate
        date = "2025-04-01"
        quantity = 300
        resultado_obtenido = get_delivery_estimate(date, quantity)
        self.assertIsInstance(resultado_obtenido, str)

    def test_generate_quote(self):
        # Prueba para la función generate_quote
        item_name = "A4 paper"
        quantity = 300
        date = "2025-04-01"
        resultado_obtenido = generate_quote(item_name, quantity, date)
        self.assertIsInstance(resultado_obtenido, dict)

    def test_get_quote_history(self):
        # Prueba para la función get_quote_history
        item_names = ["A4 paper", "Glossy paper"]
        resultado_obtenido = get_quote_history(item_names)
        self.assertIsInstance(resultado_obtenido, list)

    def test_fulfill_order(self):
        # Prueba para la función fulfill_order
        item_name = "A4 paper"
        quantity = 300
        date = "2025-04-01"
        resultado_obtenido = fulfill_order(item_name, quantity, date)
        self.assertIsInstance(resultado_obtenido, int)  # Verifica que el resultado sea un entero

if __name__ == '__main__':
    unittest.main()