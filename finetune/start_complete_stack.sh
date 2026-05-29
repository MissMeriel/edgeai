# save as: start_complete_stack.sh

#!/bin/bash

set -e

echo "================================================"
echo "Starting Complete ML Training Stack"
echo "================================================"
echo ""

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check for images
if [ ! -d "images" ] || [ -z "$(ls -A images)" ]; then
    echo "⚠️  Warning: No images found in ./images directory"
    echo "   Please add images before starting annotation."
fi

# Build and start all services
echo "Building and starting services..."
echo ""

docker-compose -f docker-compose-complete.yml up -d --build

echo ""
echo "Waiting for services to start..."
sleep 10

# Check service health
echo ""
echo "Checking service health..."
echo "---------------------------------------"

services=("fiftyone-app:5151" "streamlit-training:8501" "tensorboard:6006" "optuna-dashboard:8080")

for service in "${services[@]}"; do
    name="${service%%:*}"
    port="${service##*:}"
    
    if docker ps | grep -q "$name"; then
        echo "✓ $name is running"
    else
        echo "✗ $name failed to start"
    fi
done

# Get local IP
if command -v ip &> /dev/null; then
    LOCAL_IP=$(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v 127.0.0.1 | head -n1)
elif command -v ifconfig &> /dev/null; then
    LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print \$2}' | head -n1 | sed 's/addr://')
else
    LOCAL_IP="localhost"
fi

echo ""
echo "================================================"
echo "✅ All Services Started!"
echo "================================================"
echo ""
echo "Access your services:"
echo ""
echo "  🖼️  FiftyOne (Annotation):"
echo "      http://localhost:5151"
echo "      http://$LOCAL_IP:5151"
echo ""
echo "  🚀 Streamlit (Training):"
echo "      http://localhost:8501"
echo "      http://$LOCAL_IP:8501"
echo ""
echo "  📊 TensorBoard (Monitoring):"
echo "      http://localhost:6006"
echo "      http://$LOCAL_IP:6006"
echo ""
echo "  🔮 Optuna (AutoML):"
echo "      http://localhost:8080"
echo "      http://$LOCAL_IP:8080"
echo ""
# save as: start_complete_stack.sh (CONTINUED)

echo "  🌐 All-in-One (via Nginx):"
echo "      http://localhost"
echo "      http://$LOCAL_IP"
echo "      - /annotate  → FiftyOne"
echo "      - /train     → Streamlit"
echo "      - /tensorboard → TensorBoard"
echo "      - /automl    → Optuna"
echo ""
echo "================================================"
echo ""
echo "Useful commands:"
echo "  View logs:        docker-compose -f docker-compose-complete.yml logs -f"
echo "  Stop services:    docker-compose -f docker-compose-complete.yml down"
echo "  Restart services: docker-compose -f docker-compose-complete.yml restart"
echo "  Check status:     docker-compose -f docker-compose-complete.yml ps"
echo ""
echo "================================================"