# 🚀 Quick Start Guide

## Step 1: Install Dependencies

```bash
cd "/Users/lakshmi/Desktop/lakshmi/college/7th sem/ba/project"
pip3 install -r requirements.txt
```

## Step 2: Start the Streamlit Application

```bash
streamlit run streamlit_app.py
```

The app will automatically open in your browser at **http://localhost:8501**

---

## ✅ Verification Checklist

Before running, make sure you have:

- ✅ `Network_logs.csv` - Dataset file
- ✅ `network_logs_decision_tree_model.joblib` - Trained model (✅ Verified!)
- ✅ `streamlit_app.py` - Main application
- ✅ `preprocessing.py` - Preprocessing utilities
- ✅ All Python packages installed

---

## 📋 Project Files Status

- ✅ Model file found: `network_logs_decision_tree_model.joblib`
- ✅ Model type: DecisionTreeClassifier
- ✅ Model classes: [0=BotAttack, 1=Normal, 2=PortScan]

---

## 🎯 What You Can Do

1. **Data Overview** - View dataset statistics and distributions
2. **EDA** - Explore data with interactive visualizations
3. **Model Results** - See performance metrics for all models
4. **Inference** - Make real-time predictions on new network logs
5. **About** - Project documentation

---

## ⚠️ Troubleshooting

### If you get "ModuleNotFoundError":
```bash
pip3 install -r requirements.txt
```

### If port 8501 is already in use:
```bash
# Kill existing process
lsof -ti:8501 | xargs kill -9

# Or use a different port
streamlit run streamlit_app.py --server.port=8502
```

### If model loading fails:
- Check that `network_logs_decision_tree_model.joblib` is in the same directory
- Verify the file is not corrupted

---

## 🎉 You're Ready!

Just run: `streamlit run streamlit_app.py`


