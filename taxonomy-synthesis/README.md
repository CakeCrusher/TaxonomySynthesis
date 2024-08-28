# `taxonomy-synthesis`
An AI-driven framework for synthesizing adaptive taxonomies, enabling automated data categorization and classification within dynamic hierarchical structures.

_TLDR: copy this README and throw it into ChatGPT. It will figure things out for you. (will create a "GPT" soon)_

_Join our [Discord Community](https://discord.gg/jZSVhtwTz6) for questions, discussions, and collaboration!_

_Check out our YouTube [demo video](https://www.youtube.com/shorts/6QWXs241IEo) to see Taxonomy Synthesis in action!_

## Explain Like I'm 5 🤔
Imagine you have a big box of different animals, but you’re not sure how to group them. You know there are "Mammals" and "Reptiles," but you don’t know the smaller groups they belong to, like which mammals are more similar or which reptiles go together. This tool uses smart AI helpers to figure out those smaller groups for you, like finding out there are "Rodents" and "Primates" among the mammals, and "Lizards" and "Snakes" among the reptiles. It then helps you sort all the animals into the right groups automatically, keeping everything neatly organized!

## Features 🛠️

- **Manual and Automatic Taxonomy Generation**: Flexibly create taxonomy trees manually or automatically from arbitrary items.
- **Recursive Tree Primitives**: Utilize a tree structure that supports recursive operations, making it easy to manage hierarchical data.
- **AI-Generated Subcategories**: Automatically generate subcategories using AI models based on the context and data provided.
- **AI Classification**: Automatically classify items into appropriate categories using advanced AI models.

## Quickstart Guide ([colab](https://colab.research.google.com/drive/1BgUYdeT6aP23nYm2zopjLNKAB_9-7Hp2?usp=sharing)) 🚀


In this quickstart, we'll walk you through the process of using `taxonomy-synthesis` to create a simplified phylogenetic tree for a list of animals. We'll demonstrate how to initialize the package, set up an OpenAI client, manually create a taxonomy tree, generate subcategories automatically, and classify items using AI.

### 1. Download and Install the Package

First, ensure you have the package installed. You can install taxonomy-synthesis directly using pip:

```bash
pip install taxonomy-synthesis
```

### 2. Set Up OpenAI Client

Before proceeding, make sure you have an OpenAI API key.

```python
# Set up the OpenAI client
from openai import OpenAI

client = OpenAI(api_key='sk-...')
```

### 3. Prepare Your Data

We'll start with a list of 10 animal species, each represented with an arbitrary schema containing fields like `name`, `fun fact`, `lifespan`, and `emoji`. The only required field is `id`, which should be unique for each item.

```python
# Prepare a list of items (animals) with various attributes
items = [
  {"id": "🦘", "name": "Kangaroo", "fun_fact": "Can hop at high speeds", "lifespan_years": 23, "emoji": "🦘"},
  {"id": "🐨", "name": "Koala", "fun_fact": "Sleeps up to 22 hours a day", "lifespan_years": 18, "emoji": "🐨"},
  {"id": "🐘", "name": "Elephant", "fun_fact": "Largest land animal", "lifespan_years": 60, "emoji": "🐘"},
  {"id": "🐕", "name": "Dog", "fun_fact": "Best friend of humans", "lifespan_years": 15, "emoji": "🐕"},
  {"id": "🐄", "name": "Cow", "fun_fact": "Gives milk", "lifespan_years": 20, "emoji": "🐄"},
  {"id": "🐁", "name": "Mouse", "fun_fact": "Can squeeze through tiny gaps", "lifespan_years": 2, "emoji": "🐁"},
  {"id": "🐊", "name": "Crocodile", "fun_fact": "Lives in water and land", "lifespan_years": 70, "emoji": "🐊"},
  {"id": "🐍", "name": "Snake", "fun_fact": "No legs", "lifespan_years": 9, "emoji": "🐍"},
  {"id": "🐢", "name": "Turtle", "fun_fact": "Can live over 100 years", "lifespan_years": 100, "emoji": "🐢"},
  {"id": "🦎", "name": "Gecko", "fun_fact": "Can climb walls", "lifespan_years": 5, "emoji": "🦎"}
]
```

### 4. Initialize the Tree Structure

Create the root node for our taxonomy tree and initialize two subclasses: `Mammals` and `Reptiles`.

```python
from taxonomy_synthesis.models import Category, Item
from taxonomy_synthesis.tree.tree_node import TreeNode

# Create root node and two primary subclasses
root_category = Category(name="Animals", description="All animals")
mammal_category = Category(name="Mammals", description="Mammal species")
reptile_category = Category(name="Reptiles", description="Reptile species")

root_node = TreeNode(value=root_category)
mammal_node = TreeNode(value=mammal_category)
reptile_node = TreeNode(value=reptile_category)

# Add subclasses to the root node
root_node.add_child(mammal_node)
root_node.add_child(reptile_node)
```

### 5. Classify Items in the Root Node

Classify all items under the root node into `Mammals` or `Reptiles` using the AI classifier.

```python
from taxonomy_synthesis.tree.node_operator import NodeOperator
from taxonomy_synthesis.classifiers.gpt_classifier import GPTClassifier

# Initialize the GPT classifier and node operator
classifier = GPTClassifier(client=client)
generator = None  # We'll use manual generation for this part
operator = NodeOperator(classifier=classifier, generator=generator)

# Convert dictionary items to Item objects and classify
item_objects = [Item(**item) for item in items]
classified_items = operator.classify_items(root_node, item_objects)

print("After initial classification:")
print(root_node.print_tree())
```
_Output:_
```
After initial classification:
Animals: []
  Mammals: [🦘, 🐨, 🐘, 🐕, 🐄, 🐁]
  Reptiles: [🐊, 🐍, 🐢, 🦎]
```

### 6. Generate Subcategories for Mammals

Use AI to automatically generate subcategories under `Mammals` based on the provided data.

```python
from taxonomy_synthesis.generator.taxonomy_generator import TaxonomyGenerator

# Initialize the Taxonomy Generator
generator = TaxonomyGenerator(
    client=client,
    max_categories=2,
    generation_method="Create categories inaccordance to the philogenetic tree."
)
operator.generator = generator

# Generate subcategories under Mammals
new_categories = operator.generate_subcategories(mammal_node)

print("Generated subcategories under 'Mammals':")
print(mammal_node.print_tree())
```
_Output:_
```
Generated subcategories under 'Mammals':
Mammals: [🦘, 🐨, 🐘, 🐕, 🐄, 🐁]
  marsupials: []
  placentals: []
```

### 7. Reclassify Items under Mammals

Now classify the items specifically under the `Mammals` node into their newly generated subcategories.

```python
# Reclassify items under Mammals based on the new subcategories
classified_items = operator.classify_items(mammal_node, mammal_node.get_all_items())

print("After reclassification under 'Mammals':")
print(root_node.print_tree())
```
_Output:_
```
After reclassification under 'Mammals':
Mammals: []
  marsupials: [🦘, 🐨]
  placentals: [🐘, 🐕, 🐄, 🐁]
```

### 8. Print the Final Tree Structure

Finally, print the entire tree to see the categorized structure.

```python
# Print the final tree structure
print("Final taxonomy tree structure:")
print(root_node.print_tree())
```
_Output:_
```
Final taxonomy tree structure:
Animals: []
  Mammals: []
    marsupials: [🦘, 🐨]
    placentals: [🐘, 🐕, 🐄, 🐁]
  Reptiles: [🐊, 🐍, 🐢, 🦎]
```

## System Diagram 🎨

For a visual representation of the system architecture and its components, refer to the following diagram:

![v1 Class Diagram](https://github.com/user-attachments/assets/ffdbe2b1-4ad4-4b2b-9a72-5b14b2f3adfa)

## Contributing 🤗

Contributions are welcome! To get started, follow these steps to set up your development environment:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/CakeCrusher/TaxonomySynthesis.git
   cd taxonomy-synthesis
   ```

2. **Install Poetry** (if not already installed):

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install Dependencies**:

   Use Poetry to install all the dependencies in a virtual environment:

   ```bash
   poetry install
   ```

4. **Activate the Virtual Environment**:

   To activate the virtual environment created by Poetry:

   ```bash
   poetry shell
   ```

5. **Run Pre-Commit Hooks**:

   To maintain code quality, please run pre-commit hooks before submitting any pull requests:

   ```bash
   poetry run pre-commit install
   poetry run pre-commit run --all-files
   ```

We encourage you to open issues for any bugs you encounter or features you'd like to see added. Pull requests are also highly appreciated! Let's work together to improve and expand this project.