from pathlib import Path
import time
from model_utils import load_model, predict
from PIL import Image
import streamlit as st
import torch

st.set_page_config(
    page_title="The Horizon — Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== Warm Light Theme CSS (Dark Mode Override Safe) =====
st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FAF7F2 !important;
        color: #292524 !important;
    }
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
        background-color: #F3ECE0 !important;
        border-right: 1px solid #E2D5C1 !important;
        color: #292524 !important;
    }
    h1, h2, h3, h4, p, span, label, div, li, small {
        color: #292524;
    }
    .hero-box {
        background: linear-gradient(120deg, #FFFBEB 0%, #FEF3C7 100%) !important;
        border: 1px solid #FDE68A !important;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.2rem;
    }

    /* Result Card Container */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border-color: #E7E5E4 !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 8px rgba(41, 37, 36, 0.04);
    }

    /* File Uploader Label, Dropzone & Button */
    div[data-testid="stFileUploader"] label {
        color: #451A03 !important;
        font-weight: 600 !important;
    }
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #FFFBEB !important;
        border: 2px dashed #D97706 !important;
        border-radius: 14px !important;
        color: #292524 !important;
    }
    section[data-testid="stFileUploaderDropzone"]:hover {
        background-color: #FEF3C7 !important;
        border-color: #B45309 !important;
    }
    section[data-testid="stFileUploaderDropzone"] small,
    section[data-testid="stFileUploaderDropzone"] span,
    section[data-testid="stFileUploaderDropzone"] div {
        color: #78350F !important;
        opacity: 1 !important;
    }
    section[data-testid="stFileUploaderDropzone"] button,
    [data-testid="stBaseButton-secondary"] {
        background-color: #FEF3C7 !important;
        color: #451A03 !important;
        border: 1.5px solid #D97706 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    section[data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stBaseButton-secondary"]:hover {
        background-color: #FDE68A !important;
    }

    /* Tooltip */
    [data-testid="stTooltipContent"] {
        background-color: #FFFBEB !important;
        color: #451A03 !important;
        border: 1px solid #F59E0B !important;
    }

    /* Selectbox / Dropdown / Expander */
    div[data-baseweb="select"] > div,
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li {
        background-color: #FFFFFF !important;
        color: #292524 !important;
        border-color: #D6C7B2 !important;
    }
    div[data-testid="stExpander"],
    div[data-testid="stExpander"] details,
    div[data-testid="stExpander"] summary {
        background-color: #FFFFFF !important;
        color: #292524 !important;
        border-color: #E7E5E4 !important;
    }

    /* Alert / Info Box (st.info) */
    div[data-testid="stAlert"],
    div[data-testid="stAlert"] > div,
    div[data-testid="stAlert"] > div > div {
        background-color: #FEF3C7 !important;
        color: #451A03 !important;
        border-color: #F59E0B !important;
    }
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] div {
        color: #451A03 !important;
    }

    /* Prediction Metric (st.metric) */
    div[data-testid="stMetricValue"] {
        color: #B45309 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #78350F !important;
        font-weight: 600 !important;
    }

    /* Progress Bar (st.progress) */
    div[data-testid="stProgress"] > div > div > div {
        background: linear-gradient(90deg, #F59E0B, #D97706) !important;
    }
    div[data-testid="stProgress"] > div > div {
        background-color: #FEF3C7 !important;
    }

    /* Primary Button */
    .stButton > button[kind="primary"] {
        background-color: #D97706 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.65rem 1.2rem;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #B45309 !important;
    }

    div[data-testid="stImage"] img {
        border-radius: 14px;
        border: 1px solid #E7E5E4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===== Bilingual Text Dictionary =====
TEXT = {
    "en": {
        "subtitle": (
            "Upload a waste image to classify it across 9 categories using"
            " trained deep learning models."
        ),
        "settings": "Settings",
        "language": "Language / ဘာသာစကား",
        "mode": "Evaluation Mode",
        "single_mode": "Single Model",
        "compare_mode": "Compare All 3 Models",
        "select_model": "Select Model",
        "uploader": "Choose an image (JPG, JPEG, PNG)",
        "uploaded_caption": "Uploaded Image",
        "predict_btn": "Run Prediction",
        "idle_info": (
            "Upload a photo and click **Run Prediction** to view classification"
            " results."
        ),
        "top_pred": "Top Prediction",
        "confidence": "Confidence",
        "top3_title": "Top-3 Predictions",
        "tip_prefix": "Disposal Tip",
        "missing_file": "Checkpoint file not found",
        "ethics_title": "📌 Ethics & Educational Disclaimer",
        "ethics_body": """
* **Educational Purpose Only:** This application is created strictly for **educational and learning purposes** as a computer vision project comparing `Custom CNN`, `MobileNet V2`, and `ResNet50`.
* **Model Limitations:** Predictions are probabilistic estimates based on training data and may make mistakes on poor lighting, unusual angles, or mixed waste items. Do not rely solely on this tool for industrial, environmental, or safety-critical decisions.
* **Privacy & Data Handling:** Uploaded images are processed temporarily in memory for inference only and are not stored permanently. Please avoid uploading private or sensitive photos.
        """,
        "footer": (
            "⚠️ Built for educational & demonstration purposes only. Always"
            " follow local waste-sorting guidelines."
        ),
    },
    "my": {
        "subtitle": (
            "လေ့ကျင့်ထားသော Deep Learning Model များဖြင့် အမှိုက်ပုံများကို အမျိုးအစား ၉"
            " မျိုးခွဲခြားစစ်ဆေးနိုင်ပါသည်။"
        ),
        "settings": "ဆက်တင်များ (Settings)",
        "language": "Language / ဘာသာစကား",
        "mode": "စစ်ဆေးမည့် ပုံစံ",
        "single_mode": "Model တစ်ခုတည်း စစ်မည်",
        "compare_mode": "Model ၃ ခုလုံး ယှဉ်စစ်မည်",
        "select_model": "Model ရွေးချယ်ပါ",
        "uploader": "အမှိုက်ပုံတစ်ပုံ ရွေးချယ်တင်ပါ (JPG, JPEG, PNG)",
        "uploaded_caption": "တင်ထားသော ပုံ",
        "predict_btn": "ခန့်မှန်းချက် စစ်ဆေးမည် (Predict)",
        "idle_info": (
            "ဘယ်ဘက်တွင် ပုံတင်ပြီး **ခန့်မှန်းချက် စစ်ဆေးမည်** ခလုတ်ကို နှိပ်ပါ။"
        ),
        "top_pred": "အဓိက ခန့်မှန်းချက်",
        "confidence": "ယုံကြည်မှုနှုန်း",
        "top3_title": "ဖြစ်နိုင်ခြေ အများဆုံး အဆင့် (၃) ခု",
        "tip_prefix": "စွန့်ပစ်ရန် အကြံပြုချက်",
        "missing_file": "Checkpoint ဖိုင် ရှာမတွေ့ပါ",
        "ethics_title": "📌 Ethics နှင့် ပညာရေးဆိုင်ရာ အသိပေးချက် (Educational Disclaimer)",
        "ethics_body": """
* **ပညာရေးအတွက်သာ ဖြစ်ပါသည် (Educational Purpose Only):** ဤ application သည် Computer Vision နှင့် Machine Learning လေ့လာသင်ယူမှု (Educational & Demo) အတွက်သာ ရည်ရွယ်တည်ဆောက်ထားခြင်း ဖြစ်ပါသည်။
* **Model ၏ ကန့်သတ်ချက်များ:** ရလဒ်များသည် training data ပေါ်မူတည်၍ ခန့်မှန်းပေးခြင်းသာဖြစ်ရာ အလင်းအမှောင် အနေအထား၊ ပုံရိုက်ထားသည့် ထောင့်နှင့် ရောနှောနေသော အမှိုက်များတွင် မှားယွင်းနိုင်ပါသည်။ စက်မှုလုပ်ငန်း သို့မဟုတ် အရေးကြီး ဆုံးဖြတ်ချက်များအတွက် တစ်ခုတည်း အားကိုးအသုံးမပြုသင့်ပါ။
* **ကိုယ်ရေးအချက်အလက် လုံခြုံမှု (Privacy):** တင်လိုက်သော ပုံများကို ခန့်မှန်းချက်ထုတ်ရန်အတွက်သာ ယာယီအသုံးပြုပြီး သိမ်းဆည်းထားခြင်း မရှိပါ။ ကိုယ်ရေးကိုယ်တာ အရေးကြီးပုံများ မတင်မိစေရန် သတိပြုပါ။
        """,
        "footer": (
            "⚠️ ပညာရေးနှင့် လေ့လာစမ်းသပ်မှု အတွက်သာ ဖြစ်ပါသည်။ သက်ဆိုင်ရာ ဒေသတွင်း"
            " အမှိုက်ခွဲခြားစွန့်ပစ်မှု စည်းကမ်းများကို လိုက်နာပါ။"
        ),
    },
}

CLASS_LABELS_MY = {
    "Cardboard": "ကတ်ထူစက္ကူ",
    "Food Organics": "စားကြွင်းစားကျန်/သဘာဝအမှိုက်",
    "Glass": "ဖန်ထည်",
    "Metal": "သတ္တု",
    "Miscellaneous Trash": "အထွေထွေအမှိုက်",
    "Paper": "စက္ကူ",
    "Plastic": "ပလတ်စတစ်",
    "Textile Trash": "အဝတ်အထည်အမှိုက်",
    "Vegetation": "သစ်ရွက်/ဥယျာဉ်အမှိုက်",
}

WASTE_TIPS = {
    "en": {
        "Cardboard": "Flatten boxes and keep them dry before recycling.",
        "Food Organics": "Suitable for composting or organic waste bins.",
        "Glass": "Rinse gently and separate by color if required locally.",
        "Metal": "Rinse cans/tins and check local metal recycling rules.",
        "Miscellaneous Trash": (
            "Non-recyclable mixed items usually go to general waste."
        ),
        "Paper": "Keep clean and dry; avoid heavily soiled or waxed paper.",
        "Plastic": "Empty and rinse containers; check resin codes locally.",
        "Textile Trash": (
            "Donate wearable items or use textile recycling drop-offs."
        ),
        "Vegetation": "Great for yard waste or compost piles.",
    },
    "my": {
        "Cardboard": "ကတ်ထူပုံးများကို ပြားအောင်ခေါက်ပြီး ခြောက်သွေ့အောင်ထား၍ ပြန်လည်အသုံးပြုပါ။",
        "Food Organics": "မြေဆွေးပြုလုပ်ရန် သို့မဟုတ် သဘာဝအမှိုက်ပုံးထဲ ထည့်ရန် သင့်လျော်ပါသည်။",
        "Glass": "ရေဖြင့် ညင်သာစွာ ဆေးကြောပြီး ဖန်အမှိုက်သီးသန့် ခွဲထားပါ။",
        "Metal": "သံဘူး/သတ္တုဘူးများကို သန့်စင်ပြီး ပြန်လည်အသုံးပြု (Recycle) နိုင်ပါသည်။",
        "Miscellaneous Trash": "ပြန်လည်အသုံးပြုရန် ခက်ခဲသဖြင့် သာမန်အမှိုက်ပုံးထဲ စွန့်ပစ်ပါ။",
        "Paper": "စက္ကူများကို သန့်ရှင်းခြောက်သွေ့အောင်ထားပါ (ဆီပေနေသော စက္ကူများ မရောပါနှင့်)။",
        "Plastic": "ပလတ်စတစ်ဘူးများကို ရေဆေးသန့်စင်ပြီး Recycle အမှိုက်အဖြစ် ခွဲထားပါ။",
        "Textile Trash": "ပြန်ဝတ်နိုင်သေးပါက လှူဒါန်းပါ၊ မရပါက အထည်အလိပ် recycle နေရာသို့ ပို့ပါ။",
        "Vegetation": "သစ်ရွက်ခြောက်နှင့် အပင်အကြွင်းအကျန်များကို မြေဆွေးအဖြစ် အသုံးပြုနိုင်ပါသည်။",
    },
}

MODEL_PATHS = {
    "Custom CNN": Path("custom_cnn_best.pth"),
    "MobileNet V2": Path("mobilenet_v2_best.pth"),
    "ResNet50": Path("resnet50_best.pth"),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def get_cached_model(ckpt_path_str: str, device: str):
  return load_model(ckpt_path_str, device=device)


# ===== Sidebar =====
with st.sidebar:
  lang_choice = st.radio(
      "Language / ဘာသာစကား",
      options=["English", "မြန်မာ"],
      horizontal=True,
  )
  lang = "en" if lang_choice == "English" else "my"
  t = TEXT[lang]

  st.divider()
  st.header(t["settings"])
  mode = st.radio(t["mode"], [t["single_mode"], t["compare_mode"]])
  selected_model_name = st.selectbox(
      t["select_model"],
      list(MODEL_PATHS.keys()),
      disabled=(mode != t["single_mode"]),
  )

# ===== Hero Banner =====
st.markdown(
    f"""
    <div class="hero-box">
        <h2 style="margin:0; color:#451A03;">♻️ THE HORIZON — Waste Classification</h2>
        <p style="margin:0.4rem 0 0 0; color:#78350F; font-size:15px;">{t["subtitle"]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===== Main Section =====
uploaded_file = st.file_uploader(t["uploader"], type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert("RGB")
  left_col, right_col = st.columns([1, 1.25], gap="large")

  with left_col:
    st.image(image, caption=t["uploaded_caption"], use_container_width=True)
    predict_clicked = st.button(
        t["predict_btn"], type="primary", use_container_width=True
    )

  with right_col:
    if not predict_clicked:
      st.info(t["idle_info"])
    else:
      targets = (
          list(MODEL_PATHS.items())
          if mode == t["compare_mode"]
          else [(selected_model_name, MODEL_PATHS[selected_model_name])]
      )

      for label, path in targets:
        if not path.exists():
          st.error(f"{t['missing_file']}: `{path}`")
          continue

        model, ckpt = get_cached_model(str(path), DEVICE)

        t0 = time.perf_counter()
        top_results = predict(model, ckpt, image, device=DEVICE, top_k=3)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        best_cls, best_prob = top_results[0]
        display_best = (
            CLASS_LABELS_MY.get(best_cls, best_cls) if lang == "my" else best_cls
        )

        with st.container(border=True):
          st.subheader(label)

          m1, m2 = st.columns(2)
          m1.metric(t["top_pred"], display_best)
          m2.metric(t["confidence"], f"{best_prob * 100:.1f}%")

          st.caption(
              f"Inference: {elapsed_ms:.0f} ms · Image Size:"
              f" {ckpt.get('image_size', 'default')}px"
          )
          st.markdown(f"**{t['top3_title']}**")

          for cls_name, prob in top_results:
            pct = float(prob) * 100
            cls_display = (
                CLASS_LABELS_MY.get(cls_name, cls_name)
                if lang == "my"
                else cls_name
            )
            st.write(f"- **{cls_display}** — {pct:.2f}%")
            st.progress(min(max(float(prob), 0.0), 1.0))

          tip = WASTE_TIPS[lang].get(best_cls)
          if tip:
            st.caption(f"💡 **{t['tip_prefix']}:** {tip}")
else:
  st.info(t["idle_info"])

# ===== Ethics & Educational Disclaimer =====
st.divider()
with st.expander(t["ethics_title"], expanded=True):
  st.markdown(t["ethics_body"])

st.caption(t["footer"])