# Simplification and Aggregation of Vector Building Footprints using Deep Learning

This project implements vector-based map generalization techniques focusing on simplifying and aggregating building footprints using Graph Deep Learning models, specifically Graph Attention Networks (GAT), Graph Convolutional Networks (GCN), and GraphSAGE.

---

## Project Structure

The project consists of three main components:

1. **Preprocessing**: Transforming original geographic data into graph structures for model input.
2. **Model & Training**: Utilizing Graph Deep Learning models to learn simplification and aggregation of building polygons.
3. **Post-processing & Evaluation**: Applying genetic algorithms and geometric operations for optimizing predictions and evaluating model performance.

---

## Requirements

Ensure you have Python and required libraries installed:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
torch
torch-geometric
numpy
pandas
matplotlib
scikit-learn
networkx
```

---

## Usage

### 1. Preprocessing

Prepare your dataset into graph structures:

```bash
python preprocessing.py
```

### 2. Model & Training

Train and compare GAT, GCN, and GraphSAGE models:

```bash
python train.py --model graphsage --epochs 100
```

### 3. Post-processing & Evaluation

Evaluate your trained model and optimize results:

```bash
python evaluate.py
```

---

## Project Workflow

```
Original Vector Data → Graph Transformation → Graph Deep Learning → Result Optimization → Evaluation
```

---

## Results

GraphSAGE achieved the best performance:
- **Link Prediction Accuracy**: 0.9458
- **Node Movement MSE**: 1.3276

---

## Contributing

Feel free to open issues or pull requests for suggestions or improvements.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

- Yanning Wang
- Email: wyanning@outlook.com


