# save as: stop_stack.sh

#!/bin/bash

echo "Stopping ML Training Stack..."
echo "=============================="

docker-compose -f docker-compose-complete.yml down

echo ""
echo "✓ All services stopped"
echo ""
echo "To remove volumes as well (deletes data!):"
echo "  docker-compose -f docker-compose-complete.yml down -v"