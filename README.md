# Multimodal Search Engine 

A Python-based AI project that allows **searching images by text** and **text by image**, using pre-trained models like CLIP.

---

##  Features

- Text-to-image and image-to-text search
- Uses OpenAI CLIP model
- Cosine similarity for ranking results
- Fast search across a custom dataset
- Extensible design for larger-scale deployment
- Streamlit-based interactive UI

---

##  Technologies Used

- Python
- [CLIP (OpenAI)](https://openai.com/research/clip)
- [BLIP Image Captioning](https://huggingface.co/Salesforce/blip-image-captioning-base)
- NumPy
- PIL
- scikit-learn
- Streamlit

---

##  Dataset

We used the **Flickr30k** dataset for the first steps.
Then we used a **small subset of Flickr images** for testing to keep it lightweight.

**Download the small dataset here:**  
[Small Flickr Data for Image Captioning (Kaggle)](https://www.kaggle.com/datasets/keenwarrior/small-flicker-data-for-image-captioning)

⚠️ ⚠️ If you want to work with the full Flickr30k dataset, it is much larger (~1GB+) and may be slow on low-spec machines. 
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

# Run the Streamlit app
streamlit run app.py
```

##  Author
- Serine Lar
- MSc. Intelligent Computer Systems Engineering
- Contact: laroui.serinee@gmail.com

##  License
This project is licensed under the MIT License.