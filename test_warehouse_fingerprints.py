import json
import unittest
from decimal import Decimal

from warehouse_fingerprints import jsonb_text


class FingerprintTests(unittest.TestCase):
    def test_numeric_scale_is_not_content(self):
        left = json.loads('{"a":7.0,"b":-0.0,"c":1.2000}', parse_float=Decimal)
        right = json.loads('{"a":7,"b":0,"c":1.2}', parse_float=Decimal)
        self.assertEqual(jsonb_text(left), jsonb_text(right))

    def test_no_rounding_of_small_numbers(self):
        self.assertEqual(jsonb_text(Decimal('1.2300e-25')), '0.000000000000000000000000123')
        self.assertNotEqual(jsonb_text(Decimal('1.00000000000000001')), '1')

    def test_postgres_key_order_and_nested_values(self):
        self.assertEqual(jsonb_text({'bb': None, 'c': [True, Decimal('2.0')], 'a': 'quote"'}),
                         '{"a": "quote\\\"", "c": [true, 2], "bb": null}')
