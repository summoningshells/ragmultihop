#!/bin/bash

# Unified deployment script for GreenPower RAG Hybrid System
# This script handles complete installation and initialization

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     GreenPower RAG Hybrid System - Deployment Script              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_step() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo -e "${BLUE}$1${NC}"
    echo "═══════════════════════════════════════════════════════════════════"
}

# Step 1: Check Python version
print_step "1️⃣  CHECKING PYTHON VERSION"
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Python $PYTHON_VERSION found"

# Step 2: Check pip
print_step "2️⃣  CHECKING PIP"
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    print_error "pip is not installed"
    exit 1
fi
print_success "pip is available"

# Determine pip command
if command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    PIP_CMD="pip3"
fi

# Step 3: Virtual Environment
print_step "3️⃣  VIRTUAL ENVIRONMENT SETUP"
if [ -d "venv" ]; then
    print_info "Virtual environment already exists"
    read -p "Do you want to recreate it? (y/N): " recreate
    if [[ $recreate == "y" || $recreate == "Y" ]]; then
        print_info "Removing old virtual environment..."
        rm -rf venv
        print_info "Creating new virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment recreated"
    else
        print_success "Using existing virtual environment"
    fi
else
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Step 4: Upgrade pip
print_step "4️⃣  UPGRADING PIP"
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded to latest version"

# Step 5: Install dependencies
print_step "5️⃣  INSTALLING DEPENDENCIES"
print_info "Installing packages from requirements.txt..."
pip install -r requirements.txt
print_success "All dependencies installed"

# Step 6: Environment configuration
print_step "6️⃣  ENVIRONMENT CONFIGURATION"
if [ ! -f ".env" ]; then
    print_warning ".env file not found"
    if [ -f "env.example" ]; then
        print_info "Creating .env from env.example..."
        cp env.example .env
        print_warning "Please edit .env and add your API keys and credentials"
        print_info "Required variables:"
        echo "  - MISTRAL_API_KEY"
        echo "  - QDRANT_ENDPOINT"
        echo "  - QDRANT_API_KEY"
        echo "  - NEO4J_URI"
        echo "  - NEO4J_USERNAME"
        echo "  - NEO4J_PASSWORD"
        echo "  - NEO4J_DATABASE"
        echo ""
        read -p "Press Enter after you've configured .env..."
    else
        print_error "env.example not found. Please create .env manually"
        exit 1
    fi
else
    print_success ".env file exists"
fi

# Step 7: Initialize system
print_step "7️⃣  SYSTEM INITIALIZATION"
print_info "Running initialization script..."
python init_system.py

# Step 8: Summary and next steps
print_step "🎉 DEPLOYMENT COMPLETE"
echo ""
print_success "GreenPower RAG Hybrid System is ready!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  NEXT STEPS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  1. Activate the virtual environment (if not already active):"
echo "     source venv/bin/activate"
echo ""
echo "  2. Launch the application:"
echo "     streamlit run app_hybrid.py"
echo ""
echo "  3. Open your browser to:"
echo "     http://localhost:8501"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  USEFUL COMMANDS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Reload Neo4j data:     python neo4j_loader.py"
echo "  Re-initialize system:  python init_system.py"
echo "  Query Neo4j directly:  python neo4j_query.py"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
print_info "Tip: Use the 'Routeur Intelligent (Auto)' tab for best results"
echo ""
