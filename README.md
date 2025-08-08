# Multimodal Search Engine 

A Python-based AI project that allows **searching images by text** and **text by image**, using pre-trained models like CLIP.

---

##  Features

- Text-to-image and image-to-text search
- Uses OpenAI CLIP model
- Cosine similarity for ranking results
- Fast search across a custom dataset
- Extensible design for larger-scale deployment

---

##  Technologies Used

- Python
- CLIP (OpenAI)
- NumPy
- PIL
- scikit-learn
- Streamlit (optional UI)

---

##  Dataset

We used the **Flickr30k** dataset for testing.

⚠️ The dataset file is too large to be uploaded to GitHub.  
 You can [download it here](https://github.com/paperswithcode/paperswithcode-data)

##  How to Run

```bash
# Clone the repo
git clone https://github.com/serinelar/multimodal-search.git
cd multimodal-search

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

##  Author
- Serine Lar
- MSc. Intelligent Computer Systems Engineering
- Contact: laroui.serinee@gmail.com

##  License
This project is licensed under the MIT License.