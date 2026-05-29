# save as: streamlit_with_tensorboard.py

"""
Enhanced Streamlit UI with embedded TensorBoard
"""

import streamlit as st
import streamlit.components.v1 as components


def embed_tensorboard(logdir: str, port: int = 6006, height: int = 800):
    """Embed TensorBoard in Streamlit"""
    
    st.markdown("### 📊 TensorBoard - Live Training Metrics")
    
    # TensorBoard iframe
    tensorboard_url = f"http://localhost:{port}"
    
    components.iframe(tensorboard_url, height=height, scrolling=True)
    
    st.info(f"💡 TensorBoard is running at: {tensorboard_url}")


# Add to streamlit_training_ui.py Monitor page
def render_training_monitor_enhanced(self):
    """Enhanced monitor with TensorBoard"""
    
    st.markdown("## 📊 Training Monitor")
    
    # Status and metrics (existing code)
    # ...
    
    # Add TensorBoard embed
    st.markdown("---")
    
    with st.expander("📊 TensorBoard (Detailed Metrics)", expanded=False):
        embed_tensorboard("training_logs")
    
    # Rest of monitoring code
    # ...