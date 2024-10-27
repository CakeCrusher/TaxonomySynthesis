from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import sys

# Force UTF-8 output encoding
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class Category:
    def __init__(self, name: str, description: str, parent=None):
        self.name = name
        self.description = description
        self.parent = parent
        self.items = []

class Item:
    def __init__(self, id: str, name: str, description: str):
        self.id = id
        self.name = name
        self.description = description

class ClassifiedItem:
    def __init__(self, item: Item, category: Category, confidence: float):
        self.item = item
        self.category = category
        self.confidence = confidence

class BertClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            
        return embeddings[0]
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return float(np.dot(embedding1, embedding2) / 
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

    def classify_items(self, items: List[Dict[str, Any]], categories: List[Category], debug=True) -> List[ClassifiedItem]:
        classified_items = []
        
        category_embeddings = {}
        for category in categories:
            category_text = f"{category.name}: {category.description}"
            category_embeddings[category] = self.encode_text(category_text)
        
        for item_dict in items:
            # Combine more information for better classification
            item_text = f"{item_dict['name']}: {item_dict['fun_fact']}. This animal has a lifespan of {item_dict['lifespan_years']} years."
            item = Item(item_dict['id'], item_dict['name'], item_dict['fun_fact'])
            
            item_embedding = self.encode_text(item_text)
            
            if debug:
                print(f"\nClassifying {item_dict['name']}:")
                for category in categories:
                    similarity = self.compute_similarity(item_embedding, category_embeddings[category])
                    print(f"Similarity with {category.name}: {similarity:.3f}")
            
            best_category = None
            best_similarity = -1
            
            for category, category_embedding in category_embeddings.items():
                similarity = self.compute_similarity(item_embedding, category_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_category = category
            
            classified_item = ClassifiedItem(item, best_category, best_similarity)
            classified_items.append(classified_item)
            best_category.items.append(item)
            
        return classified_items

def classify_items_example():
    # Create categories with more detailed descriptions
    mammals = Category(
        "Mammals",
        """Warm-blooded animals that give birth to live young. They have fur or hair, 
        produce milk for their babies, and breathe air with lungs. Examples include dogs, 
        cats, elephants, whales, kangaroos, and mice. Mammals typically care for their young,
        have more developed brains, and maintain a constant body temperature."""
    )
    
    reptiles = Category(
        "Reptiles",
        """Cold-blooded animals that typically lay eggs and have scales. They are unable 
        to regulate their own body temperature and need to bask in the sun to warm up. 
        Examples include snakes, lizards, crocodiles, and turtles. Reptiles often have 
        dry, scaly skin, and most species lay eggs."""
    )
    
    categories = [mammals, reptiles]
    
    # Sample items
    items = [
        {"id": "kangaroo", "name": "Kangaroo", "fun_fact": "Can hop at high speeds and carries babies in a pouch", "lifespan_years": 23, "emoji": "🦘"},
        {"id": "koala", "name": "Koala", "fun_fact": "Sleeps up to 22 hours a day and feeds on eucalyptus leaves", "lifespan_years": 18, "emoji": "🐨"},
        {"id": "elephant", "name": "Elephant", "fun_fact": "Largest land animal, has a trunk and tusks", "lifespan_years": 60, "emoji": "🐘"},
        {"id": "dog", "name": "Dog", "fun_fact": "Best friend of humans, can be trained, gives birth to puppies", "lifespan_years": 15, "emoji": "🐕"},
        {"id": "cow", "name": "Cow", "fun_fact": "Gives milk and has multiple stomachs", "lifespan_years": 20, "emoji": "🐄"},
        {"id": "mouse", "name": "Mouse", "fun_fact": "Can squeeze through tiny gaps, has fur", "lifespan_years": 2, "emoji": "🐁"},
        {"id": "crocodile", "name": "Crocodile", "fun_fact": "Lives in water and land, has scales and lays eggs", "lifespan_years": 70, "emoji": "🐊"},
        {"id": "snake", "name": "Snake", "fun_fact": "No legs, has scales and sheds skin", "lifespan_years": 9, "emoji": "🐍"},
        {"id": "turtle", "name": "Turtle", "fun_fact": "Has a shell, lays eggs, and is cold-blooded", "lifespan_years": 100, "emoji": "🐢"},
        {"id": "gecko", "name": "Gecko", "fun_fact": "Can climb walls, has scales, and is cold-blooded", "lifespan_years": 5, "emoji": "🦎"}
    ]
    
    # Create classifier and classify items
    classifier = BertClassifier()
    classified_items = classifier.classify_items(items, categories)
    
    # Print results
    print("\nFinal Classifications:")
    for category in categories:
        items_str = ', '.join(item.name for item in category.items)
        print(f"\n{category.name}: [{items_str}]")
    
    return classified_items

if __name__ == "__main__":
    classify_items_example()
