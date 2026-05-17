"""
Curated knowledge base entries.
These are summaries of related works (paraphrased from public research)
that complement the main AgriEngineering paper.
The RAG system embeds these alongside the PDF chunks.
"""

RELATED_WORKS = [
    {
        "id": "rw_01",
        "title": "AI Agents vs Agentic AI: A Conceptual Taxonomy",
        "summary": (
            "Agentic AI systems differ fundamentally from single AI agents. "
            "Traditional AI agents are goal-oriented single-entity systems, "
            "while agentic AI involves multiple coordinating agents that handle "
            "interrelated subtasks dynamically. This taxonomy is critical for "
            "designing systems that must operate autonomously across complex, "
            "multi-modal field environments. Multi-agent coordination reduces "
            "bottlenecks and improves fault tolerance in real-world deployments."
        ),
        "relevance": "architecture, multi-agent, agentic AI definition",
    },
    {
        "id": "rw_02",
        "title": "Vision Transformers for Agricultural Disease Detection",
        "summary": (
            "Vision Transformers (ViT) split images into fixed 16x16 patches and "
            "process them as token sequences through self-attention mechanisms. "
            "Unlike CNNs, ViTs capture long-range dependencies without locality "
            "bias, making them superior for detecting subtle disease patterns in "
            "crop images. MobileViT extends this with a hybrid CNN-transformer "
            "architecture achieving high accuracy at low computational cost, "
            "suitable for edge deployment."
        ),
        "relevance": "ViT, MobileViT, computer vision, disease detection",
    },
    {
        "id": "rw_03",
        "title": "LSTM and GRU Models for IoT Sensor Time-Series Forecasting",
        "summary": (
            "Long Short-Term Memory (LSTM) networks address vanishing gradients "
            "in RNNs through gating mechanisms (input, forget, output gates). "
            "Gated Recurrent Units (GRU) simplify this with update and reset "
            "gates, achieving comparable performance at lower computational cost. "
            "Both are widely used for forecasting soil nutrients, temperature, "
            "and humidity from IoT sensor streams. 1D-CNNs provide a "
            "non-sequential alternative by extracting local temporal features "
            "through convolution, often outperforming RNNs on shorter sequences."
        ),
        "relevance": "LSTM, GRU, 1D-CNN, time-series, soil, weather prediction",
    },
    {
        "id": "rw_04",
        "title": "LLMs in Precision Agriculture: Decision Support Systems",
        "summary": (
            "Large Language Models integrated into agricultural decision support "
            "systems enable farmers to query complex sensor data in natural language. "
            "LLMs act as domain-specific search engines returning expert-level "
            "agronomic advice. When combined with RAG pipelines, they ground "
            "responses in real-time field data rather than hallucinating. "
            "Multilingual capability is critical for deployment in developing "
            "regions where farmers may not read English."
        ),
        "relevance": "LLM, RAG, chatbot, farmer interaction, multilingual",
    },
    {
        "id": "rw_05",
        "title": "Rice Disease Classification: Deep Learning Benchmarks",
        "summary": (
            "Rice diseases including bacterial leaf blight, brown spot, leaf blast, "
            "sheath blight, and tungro cause significant yield losses globally. "
            "Deep learning models trained on annotated image datasets achieve "
            "classification accuracies exceeding 95% under controlled conditions. "
            "Key challenges include class imbalance, environmental variability "
            "in field imagery, and limited labeled data for rare disease classes. "
            "Ensemble voting across multiple models improves robustness over "
            "single-model deployment."
        ),
        "relevance": "rice disease, classification, deep learning, dataset",
    },
    {
        "id": "rw_06",
        "title": "Diffusion Models as Feature Extractors for Classification Tasks",
        "summary": (
            "Pre-trained Variational Autoencoders from latent diffusion models "
            "(e.g. Stable Diffusion v1.5) encode images into semantically rich "
            "latent representations. These latents can replace CNN feature "
            "extraction in classification pipelines, providing richer "
            "representations learned from massive image datasets. This approach "
            "enables high-quality feature extraction without task-specific "
            "pretraining, particularly useful when labeled domain data is scarce."
        ),
        "relevance": "diffusion model, VAE, feature extraction, RiceNet",
    },
    {
        "id": "rw_07",
        "title": "NPK Soil Sensing and Smart Irrigation Systems",
        "summary": (
            "Soil nitrogen (N), phosphorus (P), and potassium (K) directly "
            "determine crop yield and plant health. IoT-based NPK sensors "
            "transmit real-time nutrient readings to cloud platforms, enabling "
            "precision fertilizer application. Combined with soil moisture "
            "sensors, these systems automate irrigation decisions, reducing "
            "water waste by up to 40% while maintaining optimal growth conditions. "
            "Predictive models trained on historical sensor data further improve "
            "timing of interventions."
        ),
        "relevance": "NPK sensor, soil moisture, irrigation, IoT, precision farming",
    },
]


def get_knowledge_texts():
    """Return list of (text, metadata) tuples for embedding."""
    docs = []
    for rw in RELATED_WORKS:
        text = f"[Related Work: {rw['title']}]\n\n{rw['summary']}"
        meta = {"source": "related_work", "title": rw["title"], "id": rw["id"]}
        docs.append((text, meta))
    return docs
