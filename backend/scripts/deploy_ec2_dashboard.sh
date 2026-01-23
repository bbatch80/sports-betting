#!/bin/bash
# =============================================================================
# EC2 Dashboard Deployment Script
#
# Run this script on a fresh Ubuntu 22.04 EC2 instance to deploy the
# Streamlit dashboard with production-like performance.
#
# Prerequisites:
#   - EC2 t3.micro instance (free tier) with Ubuntu 22.04
#   - Security group allowing inbound ports 22 (SSH) and 8501 (Streamlit)
#   - Instance must be able to reach RDS (same VPC or security group rules)
#
# Usage:
#   1. SSH into your EC2 instance
#   2. Copy this script or clone the repo
#   3. Run: bash deploy_ec2_dashboard.sh
# =============================================================================

set -e  # Exit on any error

echo "=============================================="
echo "  Streamlit Dashboard EC2 Deployment"
echo "=============================================="

# Update system
echo ""
echo "[1/7] Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python and pip
echo ""
echo "[2/7] Installing Python 3.11 and pip..."
sudo apt-get install -y python3.11 python3.11-venv python3-pip git

# Clone repository (if not already present)
echo ""
echo "[3/7] Setting up project directory..."
cd ~
if [ ! -d "sports-betting" ]; then
    echo "Cloning repository..."
    git clone https://github.com/bbatch80/sports-betting.git
else
    echo "Repository exists, pulling latest..."
    cd sports-betting && git pull && cd ~
fi

# Create virtual environment
echo ""
echo "[4/7] Creating Python virtual environment..."
cd ~/sports-betting/backend
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
echo ""
echo "[5/7] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create environment file
echo ""
echo "[6/7] Setting up environment variables..."
if [ ! -f .env ]; then
    cat > .env << 'ENVFILE'
# Database connection - UPDATE THESE VALUES
DATABASE_URL=postgresql://username:password@your-rds-endpoint:5432/sports_betting

# Optional: AWS credentials (if not using instance role)
# AWS_ACCESS_KEY_ID=your_key
# AWS_SECRET_ACCESS_KEY=your_secret
# AWS_DEFAULT_REGION=us-east-1
ENVFILE
    echo ""
    echo "  ⚠️  IMPORTANT: Edit ~/sports-betting/backend/.env with your RDS credentials"
    echo "     Run: nano ~/sports-betting/backend/.env"
    echo ""
fi

# Create systemd service for Streamlit
echo ""
echo "[7/7] Creating systemd service..."
sudo tee /etc/systemd/system/streamlit-dashboard.service > /dev/null << 'SERVICE'
[Unit]
Description=Streamlit Sports Betting Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/sports-betting/backend
Environment="PATH=/home/ubuntu/sports-betting/backend/venv/bin"
ExecStart=/home/ubuntu/sports-betting/backend/venv/bin/streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
SERVICE

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable streamlit-dashboard

echo ""
echo "=============================================="
echo "  Deployment Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Edit the .env file with your RDS credentials:"
echo "   nano ~/sports-betting/backend/.env"
echo ""
echo "2. Start the dashboard:"
echo "   sudo systemctl start streamlit-dashboard"
echo ""
echo "3. Check status:"
echo "   sudo systemctl status streamlit-dashboard"
echo ""
echo "4. View logs:"
echo "   sudo journalctl -u streamlit-dashboard -f"
echo ""
echo "5. Access the dashboard at:"
echo "   http://<your-ec2-public-ip>:8501"
echo ""
echo "To restart after code changes:"
echo "   cd ~/sports-betting && git pull"
echo "   sudo systemctl restart streamlit-dashboard"
echo ""
