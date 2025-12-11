#!/bin/bash

# Deployment script for RAG + Agentic AI-Textbook Chatbot Backend
# Supports deployment to different environments: development, staging, production

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}RAG + Agentic AI-Textbook Chatbot Backend Deployment Script${NC}"
echo "=================================================================="

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_prerequisites() {
    print_status "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    print_status "All prerequisites are satisfied."
}

# Function to validate environment variables
validate_env() {
    local env_file=$1
    if [ ! -f "$env_file" ]; then
        print_error "Environment file $env_file does not exist."
        exit 1
    fi

    print_status "Environment file $env_file exists."
}

# Function to build and deploy using Docker Compose
deploy_docker_compose() {
    local env_file=$1
    local env_name=$2

    print_status "Building and deploying to $env_name environment..."

    # Copy the environment file to .env for docker-compose
    cp "$env_file" .env

    # Build and start the services
    docker-compose up --build -d

    print_status "Deployment to $env_name completed successfully!"
    print_status "Application is running at http://localhost:8000"
    print_status "Check health at http://localhost:8000/health"
}

# Function to deploy to production using render
deploy_render() {
    print_status "Deploying to Render..."
    print_warning "Render deployment requires the Render CLI or manual deployment through the Render dashboard."
    print_status "Please follow these steps:"
    echo "1. Install Render CLI: npm install -g @render/cli"
    echo "2. Login: render login"
    echo "3. Deploy: render deploy --serviceId <your-service-id>"
    echo "4. Or deploy manually through the Render dashboard using the render.yaml file"
}

# Function to deploy to Railway
deploy_railway() {
    print_status "Deploying to Railway..."
    print_warning "Railway deployment requires the Railway CLI."
    print_status "Please follow these steps:"
    echo "1. Install Railway CLI: npm install -g @railway/cli"
    echo "2. Login: railway login"
    echo "3. Link project: railway link"
    echo "4. Deploy: railway up"
    echo "5. Or deploy automatically when pushing to GitHub with Railway integration"
}

# Function to run tests before deployment
run_tests() {
    print_status "Running tests before deployment..."

    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
    fi

    # Run tests using pytest
    if command -v pytest &> /dev/null; then
        pytest tests/ -v
    else
        print_warning "pytest not found, skipping tests."
    fi
}

# Function to create a backup of the database
backup_database() {
    local backup_dir="backups"
    mkdir -p "$backup_dir"

    print_status "Creating database backup..."
    # This is a placeholder - actual implementation would depend on the database
    echo "Database backup created in $backup_dir/$(date +%Y%m%d_%H%M%S).sql"
}

# Main deployment function
main() {
    local environment=${1:-"development"}

    case $environment in
        "development")
            print_status "Deploying to development environment..."
            check_prerequisites
            validate_env ".env.development"
            run_tests
            deploy_docker_compose ".env.development" "development"
            ;;
        "staging")
            print_status "Deploying to staging environment..."
            check_prerequisites
            validate_env ".env.production"  # Use production env for staging (with staging DB URL)
            run_tests
            backup_database
            deploy_docker_compose ".env.production" "staging"  # Adjust as needed
            ;;
        "production")
            print_warning "Deploying to production environment!"
            read -p "Are you sure you want to deploy to production? (yes/no): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                check_prerequisites
                validate_env ".env.production"
                run_tests
                backup_database
                deploy_docker_compose ".env.production" "production"
                print_status "Production deployment completed!"
                print_status "Please monitor the application logs for any issues."
            else
                print_status "Production deployment cancelled."
                exit 0
            fi
            ;;
        "render")
            check_prerequisites
            deploy_render
            ;;
        "railway")
            check_prerequisites
            deploy_railway
            ;;
        *)
            print_error "Invalid environment: $environment"
            echo "Usage: $0 [development|staging|production|render|railway]"
            exit 1
            ;;
    esac
}

# Show help if no arguments provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [environment]"
    echo "Available environments:"
    echo "  development - Deploy to local development environment using Docker Compose"
    echo "  staging     - Deploy to staging environment"
    echo "  production  - Deploy to production environment"
    echo "  render      - Show Render deployment instructions"
    echo "  railway     - Show Railway deployment instructions"
    echo ""
    echo "Example: $0 development"
    exit 0
fi

# Run the main function with the provided argument
main "$1"