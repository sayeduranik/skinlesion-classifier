import streamlit as st
import torch
import pickle
import timm
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import io
import time
from pathlib import Path
from huggingface_hub import hf_hub_download


# ========== Page Configuration ==========
st.set_page_config(
    page_title="Skin Lesion Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ========== Custom CSS ==========
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        background-color: #f0f8ff;
        margin: 1rem 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .model-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)


# ========== Utilities ==========
def download_model_if_needed(repo_id, filename):
    """Download model from Hugging Face if not present locally"""
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename} from Hugging Face..."):
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=".",
                    local_dir_use_symlinks=False
                )
                st.success(f"✅ {filename} downloaded!")
                return True
            except Exception as e:
                st.error(f"❌ Failed to download {filename}: {str(e)}")
                return False
    return True


# ========== Global Variables ==========
@st.cache_data
def get_class_info():
    """Return class information and descriptions"""
    class_info = {
        'AK': {'name': 'Actinic Keratosis', 'description': 'Precancerous skin lesion caused by sun damage'},
        'BCC': {'name': 'Basal Cell Carcinoma', 'description': 'Most common type of skin cancer'},
        'BKL': {'name': 'Benign Keratosis', 'description': 'Non-cancerous skin growth'},
        'DF': {'name': 'Dermatofibroma', 'description': 'Benign skin tumor'},
        'MEL': {'name': 'Melanoma', 'description': 'Most serious type of skin cancer'},
        'NV': {'name': 'Nevus', 'description': 'Common mole or birthmark'},
        'SCC': {'name': 'Squamous Cell Carcinoma', 'description': 'Second most common skin cancer'},
        'VASC': {'name': 'Vascular Lesion', 'description': 'Blood vessel-related skin lesion'}
    }
    return class_info


# ========== Model Loading Functions ==========
@st.cache_resource
def load_ensemble_model():
    """Load the ensemble model with caching"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Your Hugging Face repository ID (replace with your actual username)
        repo_id = "sranik/skin-lesion-ensemble"  # <-- Replace with your Hugging Face username
        
        # Download models if needed
        model_files = [
            "ConvNeXtV2_Tiny_Merged_Pytorch.pth",
            "DeiT3_Base_Merged_Pytorch.pth",
            "Stacked_Ensemble_ConvNeXtV2_DeiT3_Complete.pth",
            "Stacked_Ensemble_ConvNeXtV2_DeiT3_MetaLearner.pkl"
        ]
        
        for filename in model_files:
            if not download_model_if_needed(repo_id, filename):
                return None, None, None, None, None
        
        # Define the StackedEnsemble class
        class StackedEnsemble(torch.nn.Module):
            def __init__(self, convnext_model, deit_model, meta_model, class_names):
                super(StackedEnsemble, self).__init__()
                self.convnext_model = convnext_model
                self.deit_model = deit_model
                self.meta_model = meta_model
                self.class_names = class_names
                
            def forward(self, x):
                with torch.no_grad():
                    convnext_probs = torch.softmax(self.convnext_model(x), dim=1).cpu().numpy()
                    deit_probs = torch.softmax(self.deit_model(x), dim=1).cpu().numpy()
                    combined_features = np.concatenate((convnext_probs, deit_probs), axis=1)
                    ensemble_probs = self.meta_model.predict_proba(combined_features)
                    return torch.tensor(ensemble_probs, dtype=torch.float32)
        
        # Try loading the complete ensemble
        model_path = "Stacked_Ensemble_ConvNeXtV2_DeiT3_Complete.pth"
        
        if os.path.exists(model_path):
            with st.spinner("Loading ensemble model... This may take a moment."):
                try:
                    ensemble_state = torch.load(model_path, map_location=device, weights_only=False)
                    model = ensemble_state['ensemble_model']
                    class_names = ensemble_state['class_names']
                    input_size = ensemble_state['input_size']
                    model.eval()
                    return model, class_names, input_size, device, "complete"
                except:
                    with torch.serialization.safe_globals([StackedEnsemble]):
                        ensemble_state = torch.load(model_path, map_location=device)
                        model = ensemble_state['ensemble_model']
                        class_names = ensemble_state['class_names']
                        input_size = ensemble_state['input_size']
                        model.eval()
                        return model, class_names, input_size, device, "complete"
        else:
            st.error(f"Model file not found: {model_path}")
            st.info("Please make sure the model file is in the same directory as this app.")
            return None, None, None, None, None
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None


@st.cache_resource
def load_individual_models():
    """Alternative: Load individual models separately"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Your Hugging Face repository ID (replace with your actual username)
        repo_id = "your-username/skin-lesion-ensemble"  # <-- Replace with your Hugging Face username
        
        # Download models if needed
        model_files = [
            "ConvNeXtV2_Tiny_Merged_Pytorch.pth",
            "DeiT3_Base_Merged_Pytorch.pth",
            "Stacked_Ensemble_ConvNeXtV2_DeiT3_MetaLearner.pkl"
        ]
        
        for filename in model_files:
            if not download_model_if_needed(repo_id, filename):
                return None, None, None, None, None, None, None
        
        # Default class names (update based on your dataset)
        class_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
        num_classes = len(class_names)
        input_size = (224, 224)
        
        with st.spinner("Loading individual models..."):
            # Load ConvNeXtV2
            convnext_model = timm.create_model('convnextv2_tiny.fcmae_ft_in1k', pretrained=False, num_classes=num_classes)
            convnext_model.load_state_dict(torch.load("ConvNeXtV2_Tiny_Merged_Pytorch.pth", map_location=device))
            convnext_model.to(device).eval()
            
            # Load DeiT
            deit_model = timm.create_model('deit3_base_patch16_224', pretrained=False, num_classes=num_classes)
            deit_model.load_state_dict(torch.load("DeiT3_Base_Merged_Pytorch.pth", map_location=device))
            deit_model.to(device).eval()
            
            # Load meta-learner
            with open("Stacked_Ensemble_ConvNeXtV2_DeiT3_MetaLearner.pkl", 'rb') as f:
                meta_model = pickle.load(f)
            
            return convnext_model, deit_model, meta_model, class_names, input_size, device, "individual"
            
    except Exception as e:
        st.error(f"Error loading individual models: {str(e)}")
        return None, None, None, None, None, None, None


# ========== Prediction Functions ==========
def predict_with_ensemble(image, model, class_names, input_size, device, model_type="complete"):
    """Make prediction with ensemble model"""
    try:
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        if model_type == "complete":
            # Use complete ensemble model
            with torch.no_grad():
                predictions = model(image_tensor)
                probabilities = torch.softmax(predictions, dim=1)
        else:
            # Use individual models
            convnext_model, deit_model, meta_model = model
            with torch.no_grad():
                convnext_probs = torch.softmax(convnext_model(image_tensor), dim=1).cpu().numpy()
                deit_probs = torch.softmax(deit_model(image_tensor), dim=1).cpu().numpy()
                combined_features = np.concatenate((convnext_probs, deit_probs), axis=1)
                ensemble_probs = meta_model.predict_proba(combined_features)
                probabilities = torch.tensor(ensemble_probs, dtype=torch.float32)
        
        # Get predictions
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item()
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities, min(3, len(class_names)))
        top_predictions = []
        for i in range(min(3, len(class_names))):
            class_idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            top_predictions.append((class_names[class_idx], prob))
        
        return class_names[predicted_class_idx], confidence, top_predictions, probabilities[0].cpu().numpy()
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, None


def get_confidence_color(confidence):
    """Return CSS class based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"


# ========== Main App ==========
def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Skin Lesion Classification</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Ensemble Model for Dermatological Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Model Information")
        
        # Model loading
        model_data = load_ensemble_model()
        if model_data[0] is None:
            st.warning("Trying alternative loading method...")
            alt_model_data = load_individual_models()
            if alt_model_data[0] is None:
                st.error("Failed to load models!")
                st.stop()
            else:
                model, class_names, input_size, device, model_type = alt_model_data[0:3], alt_model_data[3], alt_model_data[4], alt_model_data[5], alt_model_data[6]
        else:
            model, class_names, input_size, device, model_type = model_data
        
        st.success("✅ Models loaded successfully!")
        
        # Model details
        with st.expander("🔧 Technical Details"):
            st.write(f"**Device:** {device}")
            st.write(f"**Input Size:** {input_size}")
            st.write(f"**Classes:** {len(class_names)}")
            st.write(f"**Model Type:** {model_type.title()}")
        
        # Class information
        with st.expander("📚 Lesion Types"):
            class_info = get_class_info()
            for class_code in class_names:
                if class_code in class_info:
                    st.write(f"**{class_code}:** {class_info[class_code]['name']}")
                    st.write(f"*{class_info[class_code]['description']}*")
                    st.write("---")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a skin lesion image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of the skin lesion for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.info(f"📏 Image Size: {image.size[0]} x {image.size[1]} pixels")
            
    with col2:
        st.header("🎯 Prediction Results")
        
        if uploaded_file is not None:
            # Prediction button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image... Please wait."):
                    start_time = time.time()
                    
                    # Make prediction
                    pred_class, confidence, top_predictions, all_probs = predict_with_ensemble(
                        image, model, class_names, input_size, device, model_type
                    )
                    
                    end_time = time.time()
                    
                    if pred_class:
                        # Main prediction
                        confidence_class = get_confidence_color(confidence)
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>🏆 Primary Prediction</h3>
                            <h2 style="color: #1f77b4;">{pred_class}</h2>
                            <p class="{confidence_class}">Confidence: {confidence:.4f} ({confidence*100:.2f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Class description
                        class_info = get_class_info()
                        if pred_class in class_info:
                            st.info(f"**{class_info[pred_class]['name']}:** {class_info[pred_class]['description']}")
                        
                        # Top predictions
                        st.subheader("📊 Top 3 Predictions")
                        for i, (class_name, prob) in enumerate(top_predictions):
                            confidence_class = get_confidence_color(prob)
                            st.markdown(f"""
                            <div style="margin: 0.5rem 0; padding: 0.5rem; background-color: #f8f9fa; border-radius: 5px;">
                                <strong>{i+1}. {class_name}</strong>
                                <span class="{confidence_class}" style="float: right;">{prob:.4f} ({prob*100:.2f}%)</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Prediction time
                        st.success(f"⏱️ Analysis completed in {end_time - start_time:.2f} seconds")
                        
                        # Probability distribution chart
                        st.subheader("📈 Probability Distribution")
                        prob_df = {
                            'Class': class_names,
                            'Probability': all_probs
                        }
                        st.bar_chart(prob_df, x='Class', y='Probability')
                    else:
                        st.error("❌ Prediction failed. Please try again.")
        else:
            st.info("👆 Please upload an image to start the analysis.")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="model-info">
            <h4>🧠 Ensemble Model</h4>
            <p>ConvNeXtV2 Tiny + DeiT III Base</p>
            <p>Meta-learning with Logistic Regression</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-info">
            <h4>⚠️ Medical Disclaimer</h4>
            <p>This tool is for research purposes only.</p>
            <p>Always consult healthcare professionals for medical advice.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="model-info">
            <h4>🔬 Dataset</h4>
            <p>Trained on dermatological image dataset</p>
            <p>8 different lesion types</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
