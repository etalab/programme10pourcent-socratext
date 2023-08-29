"""
Mappings.
"""


label2id = {
    'item_total_price': 0,
    'code_tva': 1,
    'item_unit_price': 2,
    'total_price': 3,
    'magasin': 4,
    'item_name': 5,
    'date': 6,
    'item_quantity': 7,
    'taux_tva': 8
}

id2label = {value: key for key, value in label2id.items()}
